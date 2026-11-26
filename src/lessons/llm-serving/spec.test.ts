// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import text from '../../lib/corpora/grimm.txt?raw';
import { createRng } from '../../lib/rng';
import { Decoder, drawFrom, Gpt, gptParamCount, sampleDistribution } from '../../lib/compute/gpt';
import { encode, prepareCorpus, VOCAB } from '../../lib/corpora/grimm';
import { nodeFs } from '../../lib/nodeIo';
import {
  BigramDrafter,
  DRAFT_CFG,
  DRAFT_PARAMS,
  GptDrafter,
  TARGET_CFG,
  TARGET_PARAMS,
  buildBigramLogProbs,
  exactSingleStepMarginal,
  expectedTokensFormula,
  heldOutPrompts,
  productionCostRatio,
  runPlainTarget,
  runSpeculative,
} from './spec';
import { measureCell } from './spec.run';
import { SPEC_COST_RATIOS, SPEC_GRID, SPEC_PROMPTS } from './specResults';

const corpus = prepareCorpus(text);
const bigramLogProbs = buildBigramLogProbs(corpus.train);

async function loadModels() {
  const fs = await nodeFs();
  const load = (url: URL, cfg: typeof TARGET_CFG) => {
    const b = fs.readFileSync(url);
    const m = new Gpt(cfg);
    m.load(new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
    return m;
  };
  return {
    target: load(new URL('../../lib/compute/tinyGpt.cpu.bin', import.meta.url), TARGET_CFG),
    draft: load(new URL('./draftGpt.bin', import.meta.url), DRAFT_CFG),
  };
}

describe('llm-serving S2: real tiny GPTs, rollback, chunked-prefill equivalence & speculative exactness', () => {
  it('model sizes, 12 distinct held-out prompts, and production cost ratios match', () => {
    expect(TARGET_PARAMS).toBe(29696);
    expect(DRAFT_PARAMS).toBe(5424);
    expect(gptParamCount(TARGET_CFG)).toBe(TARGET_PARAMS);
    expect(gptParamCount(DRAFT_CFG)).toBe(DRAFT_PARAMS);

    const prompts = heldOutPrompts(corpus.text, corpus.split);
    expect(prompts).toHaveLength(12);
    expect(new Set(prompts.map((p) => p.prompt)).size).toBe(12);
    expect(prompts).toEqual(SPEC_PROMPTS);

    expect(productionCostRatio('bigram').c).toBeCloseTo(0.01, 4);
    expect(productionCostRatio('small-gpt').c).toBeCloseTo(0.1266, 3);
    expect(SPEC_COST_RATIOS['small-gpt']).toBeCloseTo(0.1266, 4);
  });

  it('Decoder rollback (dec.len = savedLen) matches Gpt.nextLogits bit-for-bit', async () => {
    const { target } = await loadModels();
    const prefix = encode('There was once a king who had ');
    const bogus = encode('xyz123');
    const realNext = encode('three')[0];

    const dec = new Decoder(target);
    dec.prefill(prefix);
    const savedLen = dec.len;

    // Step 6 drafted tokens, then roll back via a single assignment
    for (const t of bogus) dec.step(t);
    expect(dec.len).toBe(savedLen + bogus.length);
    dec.len = savedLen;

    // Continue with the true correction token and compare against fresh full forward pass
    const afterRollback = dec.step(realNext);
    const fullContext = Int32Array.from([...prefix, realNext]);
    const refLogits = target.nextLogits(fullContext);

    expect(afterRollback.length).toBe(VOCAB);
    let maxDiff = 0;
    for (let i = 0; i < VOCAB; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(afterRollback[i] - refLogits[i]));
    }
    expect(maxDiff).toBe(0);
  });

  it('chunked-prefill equivalence on the real tiny GPT: chunked Decoder logits === full prefill bit-for-bit', async () => {
    const { target } = await loadModels();
    const prompt = encode('A great king of a land far away had a daughter, ');
    expect(prompt.length).toBe(48);

    const refFull = target.nextLogits(prompt);
    const wholeDec = new Decoder(target);
    const wholeLogits = wholeDec.prefill(prompt);

    for (const chunkSize of [8, 13, 16]) {
      const chunkDec = new Decoder(target);
      let logits: Float64Array = new Float64Array(0);
      for (let off = 0; off < prompt.length; off += chunkSize) {
        const end = Math.min(prompt.length, off + chunkSize);
        for (let i = off; i < end; i++) {
          logits = chunkDec.step(prompt[i]);
        }
      }
      expect(chunkDec.len).toBe(prompt.length);
      let maxDiffFull = 0;
      let maxDiffWhole = 0;
      for (let i = 0; i < VOCAB; i++) {
        maxDiffFull = Math.max(maxDiffFull, Math.abs(logits[i] - refFull[i]));
        maxDiffWhole = Math.max(maxDiffWhole, Math.abs(logits[i] - wholeLogits[i]));
      }
      expect(maxDiffFull).toBe(0);
      expect(maxDiffWhole).toBe(0);
    }
  });

  it('greedy speculative decoding matches plain target decoding character for character (all k = 1..8, both drafters)', async () => {
    const { target, draft } = await loadModels();
    const prompt = SPEC_PROMPTS[0].prompt;
    const plain = runPlainTarget({ targetModel: target, prompt, maxNewTokens: 36, temperature: 0, seed: 1 });

    for (const k of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const rBg = runSpeculative({
        targetModel: target,
        drafter: new BigramDrafter(bigramLogProbs),
        prompt,
        maxNewTokens: 36,
        k,
        temperature: 0,
        seed: 100 + k,
      });
      expect(rBg.generatedText).toBe(plain.generatedText);

      const rGpt = runSpeculative({
        targetModel: target,
        drafter: new GptDrafter(draft),
        prompt,
        maxNewTokens: 36,
        k,
        temperature: 0,
        seed: 200 + k,
      });
      expect(rGpt.generatedText).toBe(plain.generatedText);
    }
  });

  it('exact 1-step speculative marginal equals target distribution p(x) to < 1e-12, plus Monte Carlo check', async () => {
    const { target, draft } = await loadModels();
    const promptIds = encode(SPEC_PROMPTS[3].prompt);
    const tLogits = target.nextLogits(promptIds);
    const dLogits = draft.nextLogits(promptIds);
    const bgLogits = new BigramDrafter(bigramLogProbs).prefill(promptIds);

    for (const temp of [0.7, 1.0]) {
      const p = sampleDistribution(tLogits, temp, 1);
      for (const qLogits of [dLogits, bgLogits]) {
        const q = sampleDistribution(qLogits, temp, 1);
        const marg = exactSingleStepMarginal(p, q);
        let maxErr = 0;
        for (let i = 0; i < VOCAB; i++) {
          maxErr = Math.max(maxErr, Math.abs(marg[i] - p[i]));
        }
        expect(maxErr).toBeLessThan(1e-12);
      }
    }

    // Seeded Monte Carlo over full speculative accept/reject procedure (3,000 draws at T = 1.0)
    const p = sampleDistribution(tLogits, 1.0, 1);
    const q = sampleDistribution(dLogits, 1.0, 1);
    const rng = createRng(9999);
    const N = 3000;
    const specCounts = new Float64Array(VOCAB);
    const residual = new Float64Array(VOCAB);
    let rSum = 0;
    for (let i = 0; i < VOCAB; i++) {
      const d = Math.max(0, p[i] - q[i]);
      residual[i] = d;
      rSum += d;
    }
    for (let i = 0; i < VOCAB; i++) residual[i] /= rSum;

    for (let s = 0; s < N; s++) {
      const x = drawFrom(q, rng.next());
      const ap = q[x] > 0 ? Math.min(1, p[x] / q[x]) : 0;
      const emitted = rng.next() < ap ? x : drawFrom(residual, rng.next());
      specCounts[emitted]++;
    }
    let maxMcErr = 0;
    for (let i = 0; i < VOCAB; i++) {
      maxMcErr = Math.max(maxMcErr, Math.abs(specCounts[i] / N - p[i]));
    }
    // 3,000 draws -> worst SE <= 0.5 / sqrt(3000) = 0.0091; allow 3× SE (~0.028)
    expect(maxMcErr).toBeLessThan(0.028);
  });

  it('re-verifies recorded specResults.ts cells and pins headline findings', async () => {
    expect(SPEC_GRID).toHaveLength(48);
    // Every single cell is slower than plain decoding on the CPU (< 1.0× wall-clock)
    expect(Math.max(...SPEC_GRID.map((c) => c.cpuWallSpeedup))).toBeLessThan(1.0);

    const { target, draft } = await loadModels();
    const samples: [typeof SPEC_GRID[number]['drafter'], number, number][] = [
      ['bigram', 0, 5],
      ['small-gpt', 0, 7],
      ['bigram', 1.0, 4],
      ['small-gpt', 1.0, 3],
    ];
    for (const [drafter, temperature, k] of samples) {
      const recorded = SPEC_GRID.find((c) => c.drafter === drafter && c.temperature === temperature && c.k === k)!;
      const fresh = measureCell({
        targetModel: target,
        draftModel: draft,
        bigramLogProbs,
        prompts: [...SPEC_PROMPTS],
        drafter,
        temperature,
        k,
      });
      expect(fresh.alpha).toBe(recorded.alpha);
      expect(fresh.alphaWordStart).toBe(recorded.alphaWordStart);
      expect(fresh.alphaMidWord).toBe(recorded.alphaMidWord);
      expect(fresh.tokensPerPass).toBe(recorded.tokensPerPass);
      expect(fresh.modelledSpeedup).toBe(recorded.modelledSpeedup);
    }

    // Pin honest headline numbers quoted in prose & Lab 2
    const bestGreedy = [...SPEC_GRID].filter((c) => c.temperature === 0).sort((a, b) => b.modelledSpeedup - a.modelledSpeedup)[0];
    expect(bestGreedy.drafter).toBe('bigram');
    expect(bestGreedy.k).toBe(5);
    expect(bestGreedy.modelledSpeedup).toBeCloseTo(2.30, 2);

    const bestGptGreedy = [...SPEC_GRID]
      .filter((c) => c.temperature === 0 && c.drafter === 'small-gpt')
      .sort((a, b) => b.modelledSpeedup - a.modelledSpeedup)[0];
    expect(bestGptGreedy.k).toBe(7);
    expect(bestGptGreedy.tokensPerPass).toBeCloseTo(3.82, 2);
    expect(bestGptGreedy.modelledSpeedup).toBeCloseTo(2.03, 2);
    expect(expectedTokensFormula(bestGptGreedy.alpha, 7)).toBeCloseTo(4.23, 2);

    const bestTemp1 = [...SPEC_GRID].filter((c) => c.temperature === 1.0).sort((a, b) => b.modelledSpeedup - a.modelledSpeedup)[0];
    expect(bestTemp1.drafter).toBe('bigram');
    expect(bestTemp1.k).toBe(4);
    expect(bestTemp1.modelledSpeedup).toBeCloseTo(2.28, 2);
  });
});
