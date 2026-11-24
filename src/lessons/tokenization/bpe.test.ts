import { describe, expect, it } from 'vitest';
import {
  CORPUS,
  LANGUAGE_SAMPLES,
  applyMergeToWords,
  bestMdl,
  encodeWord,
  encodeWordTrace,
  mdlStep,
  topPairs,
  train,
  utf8Stats,
  vocabIds,
  wordsOf,
} from './bpe';

describe('BPE engine on the lesson corpus', () => {
  const trained = train(CORPUS);

  it('matches the corpus counts quoted in prose', () => {
    const words = wordsOf(CORPUS);
    expect(words.reduce((s, w) => s + w.count, 0)).toBe(107);
    expect(words.length).toBe(50);
    expect(trained.initialVocabSize).toBe(22);
    expect(trained.merges.length).toBe(66);
    expect(trained.steps[0].tokens).toBe(531);
    expect(trained.steps[66].tokens).toBe(177);
    expect(trained.steps[66].vocab).toBe(88);
  });

  it('learns the expected early merges and reusable suffixes', () => {
    expect(trained.merges[0]).toEqual({ pair: ['a', 't'], merged: 'at', count: 27 });
    expect(trained.merges[1]).toEqual({ pair: ['e', '_'], merged: 'e_', count: 26 });
    expect(trained.merges[2]).toEqual({ pair: ['t', 'h'], merged: 'th', count: 24 });
    expect(trained.merges[3]).toEqual({ pair: ['th', 'e_'], merged: 'the_', count: 23 });
    expect(trained.merges[13]).toEqual({ pair: ['e', 'st_'], merged: 'est_', count: 7 });
  });

  it('tokenises strawberry, unseen words, and suffixes as quoted', () => {
    expect(encodeWord('strawberry', trained.merges, 66)).toEqual(['st', 'r', 'a', 'w', 'berr', 'y', '_']);
    expect(encodeWord('unwatched', trained.merges, 66)).toEqual(['u', 'n', 'watch', 'ed_']);
    expect(encodeWord('warmest', trained.merges, 28)).toEqual(['warm', 'est_']);
    expect(encodeWord('fastest', trained.merges, 28)).toEqual(['fa', 'st', 'est_']);
    expect(encodeWord('slowest', trained.merges, 28)).toEqual(['sl', 'o', 'w', 'est_']);
    expect(encodeWord('123456', trained.merges, 66)).toEqual(['1', '2', '3', '4', '5', '6', '_']);

    const trace = encodeWordTrace('strawberry', trained.merges, 66);
    expect(trace.tokens).toEqual(['st', 'r', 'a', 'w', 'berr', 'y', '_']);
    expect(trace.applied.length).toBeGreaterThan(0);
    expect(trace.applied[trace.applied.length - 1].result).toEqual(trace.tokens);

    const ids = vocabIds(trained.baseVocab, trained.merges, 66);
    expect(ids.size).toBe(88);
    expect(trace.tokens.every((t) => ids.has(t))).toBe(true);
  });

  it('minimum description length has an exhaustive optimum at 28 merges (≈1,806 bits)', () => {
    const best = bestMdl(trained.steps);
    expect(best.merges).toBe(28);
    expect(best.tokens).toBe(264);
    expect(best.vocab).toBe(50);
    expect(Math.round(best.totalBits)).toBe(1806);

    // Endpoints quoted in prose
    expect(Math.round(trained.steps[0].totalBits)).toBe(2368);
    expect(Math.round(trained.steps[66].totalBits)).toBe(1996);

    // Corpus-only bits fall monotonically (motivating why we include the merge table)
    for (let i = 1; i < trained.steps.length; i++) {
      expect(trained.steps[i].corpusBits).toBeLessThan(trained.steps[i - 1].corpusBits);
    }

    // Turning point: merge 28 (count 4) helps, merge 29 (count 3) hurts
    expect(trained.merges[27].count).toBe(4);
    expect(trained.merges[28].count).toBe(3);
    expect(trained.steps[28].totalBits).toBeLessThan(trained.steps[27].totalBits);
    expect(trained.steps[29].totalBits).toBeGreaterThan(trained.steps[28].totalBits);

    // Attention n² reduction ratios
    expect((trained.steps[28].tokens / trained.steps[0].tokens) ** 2).toBeCloseTo(0.247, 2);
    expect((trained.steps[66].tokens / trained.steps[0].tokens) ** 2).toBeCloseTo(0.111, 2);
  });

  it('hand-applied merges reproduce greedy BPE step for step', () => {
    let words = wordsOf(CORPUS);
    for (let k = 0; k < 28; k++) {
      const top = topPairs(words, 1)[0];
      expect(top.pair).toEqual(trained.merges[k].pair);
      words = applyMergeToWords(words, top.pair[0], top.pair[1]);
    }
    const s = mdlStep(words, 28, trained.initialVocabSize);
    expect(s.totalBits).toBeCloseTo(trained.steps[28].totalBits, 6);
  });

  it('UTF-8 byte counts match the language-tax comparison', () => {
    const byLang = Object.fromEntries(LANGUAGE_SAMPLES.map((s) => [s.lang, utf8Stats(s.text)]));
    expect(byLang.English.byteCount).toBe(28);
    expect(byLang.English.bytesPerChar).toBe(1);
    expect(byLang.Spanish.byteCount).toBe(41);
    expect(byLang.Greek.byteCount).toBe(56);
    expect(byLang.Japanese.chars).toBe(15);
    expect(byLang.Japanese.byteCount).toBe(45);
    expect(byLang.Japanese.bytesPerChar).toBe(3);
    expect(byLang.Hindi.byteCount).toBe(71);
    expect(byLang.Emoji.chars).toBe(4);
    expect(byLang.Emoji.byteCount).toBe(16);
    expect(byLang.Emoji.bytesPerChar).toBe(4);
  });
});
