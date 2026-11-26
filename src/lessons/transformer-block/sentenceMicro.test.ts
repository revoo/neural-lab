// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  crossEntropyLoss,
  TARGET_AT_LAST,
  traceSentence,
  VOCAB,
} from './sentenceMicro';

describe('4-token Transformer SentenceMicroscope (sentenceMicro.ts)', () => {
  it('enforces a strict causal mask (k > q is 0) and rows sum to 1.0 in every head', () => {
    const tr = traceSentence();
    for (const head of tr.attnWeights) {
      expect(head.length).toBe(4);
      for (let q = 0; q < 4; q++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          if (k > q) expect(head[q][k]).toBe(0);
          expect(head[q][k]).toBeGreaterThanOrEqual(0);
          sum += head[q][k];
        }
        expect(Math.abs(sum - 1)).toBeLessThan(1e-12);
      }
    }
  });

  it('softmax probabilities sum to 1 and p - y matches central finite differences on logits to < 1e-8', () => {
    const tr = traceSentence();
    const eps = 1e-6;
    for (const tok of tr.tokens) {
      const pSum = tok.probs.reduce((a, b) => a + b, 0);
      expect(Math.abs(pSum - 1)).toBeLessThan(1e-12);

      for (let i = 0; i < VOCAB.length; i++) {
        const plus = tok.logits.slice();
        const minus = tok.logits.slice();
        plus[i] += eps;
        minus[i] -= eps;
        const fd = (crossEntropyLoss(plus, tok.targetId) - crossEntropyLoss(minus, tok.targetId)) / (2 * eps);
        expect(Math.abs(tok.dLogits[i] - fd)).toBeLessThan(1e-8);
      }
    }
  });

  it('pins full-block focus and unmistakable ablation drops on "on" -> "mat"', () => {
    const full = traceSentence({ useAttention: true, useResidual: true, usePositions: true });
    const noAttn = traceSentence({ useAttention: false, useResidual: true, usePositions: true });
    const noRes = traceSentence({ useAttention: true, useResidual: false, usePositions: true });
    const noPos = traceSentence({ useAttention: true, useResidual: true, usePositions: false });

    const topOf = (probs: number[]) => VOCAB[probs.indexOf(Math.max(...probs))];

    // Head focus at position 3 ("on")
    expect(full.attnWeights[0][3][1]).toBeCloseTo(0.87, 2); // Head 1 -> "cat"
    expect(full.attnWeights[1][3][2]).toBeCloseTo(0.97, 2); // Head 2 -> "sat"

    // Full block predicts "mat" cleanly at 90.0%
    expect(full.tokens[3].probs[TARGET_AT_LAST]).toBeCloseTo(0.900, 2);
    expect(topOf(full.tokens[3].probs)).toBe('mat');

    // Attention OFF drops p("mat") sharply to 13.6% and flips top-1 away from "mat"
    expect(noAttn.tokens[3].probs[TARGET_AT_LAST]).toBeCloseTo(0.136, 2);
    expect(topOf(noAttn.tokens[3].probs)).not.toBe('mat');

    // Residual OFF drops p("mat") to 5.3% and flips top-1 away from "mat"
    expect(noRes.tokens[3].probs[TARGET_AT_LAST]).toBeCloseTo(0.053, 2);
    expect(topOf(noRes.tokens[3].probs)).not.toBe('mat');

    // Positions OFF drops p("mat") to 41.4% and flips top-1 away from "mat"
    expect(noPos.tokens[3].probs[TARGET_AT_LAST]).toBeCloseTo(0.414, 2);
    expect(topOf(noPos.tokens[3].probs)).not.toBe('mat');
  });

  it('without positional encodings, shuffling earlier tokens leaves the last token strictly invariant', () => {
    const orig = traceSentence({ useAttention: true, useResidual: true, usePositions: false, tokens: [0, 1, 2, 3] });
    const shuffled = traceSentence({ useAttention: true, useResidual: true, usePositions: false, tokens: [2, 0, 1, 3] });
    for (let i = 0; i < VOCAB.length; i++) {
      expect(Math.abs(orig.tokens[3].probs[i] - shuffled.tokens[3].probs[i])).toBeLessThan(1e-12);
    }
  });
});
