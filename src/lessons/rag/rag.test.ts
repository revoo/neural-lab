// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { DOCS, QUESTIONS } from './corpus';
import { CONCEPT_NAMES, buildIndex, chunkCorpus, cosine, evaluate, tokens } from './rag';
import { LAB_BRIEFS, labExhaustive } from './lab';

const p1 = (v: number) => (100 * v).toFixed(1);

describe('corpus and retrieval machinery', () => {
  it('sizes quoted in the lesson: 16 pages, 67 sentences, 34 questions, 38 concepts', () => {
    expect(DOCS).toHaveLength(16);
    expect(DOCS.reduce((s, d) => s + d.sentences.length, 0)).toBe(67);
    expect(QUESTIONS).toHaveLength(34);
    expect(CONCEPT_NAMES).toHaveLength(38);
    for (const q of QUESTIONS) expect(DOCS.find((d) => d.id === q.doc)!.sentences[q.sent]).toBeTruthy();
  });

  it('cosine similarity of sparse vectors', () => {
    const a = new Map([
      ['x', 1],
      ['y', 2],
    ]);
    const b = new Map([
      ['y', 2],
      ['z', 1],
    ]);
    expect(cosine(a, b)).toBeCloseTo(4 / 5, 12);
    expect(cosine(a, new Map())).toBe(0);
  });

  it('tokens drop stop words and plural s, and keep codes', () => {
    expect(tokens('What does E2 mean for the shots?')).toEqual(['e2', 'mean', 'shot']);
  });

  it('the two presets: lexical finds E2 but not the refund policy; semantic the reverse', () => {
    const idx = buildIndex(chunkCorpus(1));
    const e2 = QUESTIONS.find((q) => q.q === 'What does E2 mean?')!;
    const money = QUESTIONS.find((q) => q.q === 'How soon do I get my money back?')!;
    const top = (q: string, l: number) => idx.search(q, l)[0];
    expect(top(e2.q, 0).chunk.id).toBe('errors#1');
    expect(top(e2.q, 1).score).toBe(0);
    expect(top(money.q, 0).chunk.id).toBe('safety#1'); // matched on the word “get”
    expect(idx.search(money.q, 0).find((r) => r.chunk.id === 'returns#2')!.lex).toBe(0);
    expect(top(money.q, 1).chunk.id).toBe('returns#2');
  });
});

describe('numbers quoted in the lesson', () => {
  it('one-sentence chunks, top 3: lexical 79.4% (4 never), semantic 82.4% (2 never), hybrid 97.1%', () => {
    const lex = evaluate(1, 0, 3);
    const sem = evaluate(1, 1, 3);
    const mix = evaluate(1, 0.5, 3);
    expect(p1(lex.recall)).toBe('79.4');
    expect(lex.ranks.filter((r) => r === Infinity)).toHaveLength(4);
    expect(p1(sem.recall)).toBe('82.4');
    expect(sem.ranks.filter((r) => r === Infinity)).toHaveLength(2);
    expect(p1(mix.recall)).toBe('97.1');
    expect(mix.promptWords).toBeCloseTo(49.4, 1);
  });

  it('two-sentence chunks, top 3 find every answer for 82.3 words', () => {
    const r = evaluate(2, 0.5, 3);
    expect(r.recall).toBe(1);
    expect(r.promptWords).toBeCloseTo(82.3, 1);
  });

  it('lab: fewest words 82.3 / 33 / 109.8; no pure method finds all 34; the semantic-only misses include two error codes', () => {
    const [all, tight, vec] = LAB_BRIEFS.map(labExhaustive);
    expect([all.total, all.passing, all.bestWords]).toEqual([125, 27, 82.3]);
    expect(all.best.map((b) => [b.size, b.lambda, b.k])).toEqual([
      [2, 0.25, 3],
      [2, 0.5, 3],
    ]);
    expect([tight.passing, tight.bestWords]).toEqual([66, 33]);
    expect(tight.best).toHaveLength(1);
    expect(p1(tight.best[0].recall)).toBe('94.1');
    expect([vec.total, vec.passing, vec.bestWords]).toEqual([25, 10, 109.8]);
    for (const size of [1, 2, 3, 4, 5])
      for (const k of [1, 2, 3, 4, 5]) {
        expect(evaluate(size, 0, k).recall).toBeLessThan(1);
        const s = evaluate(size, 1, k);
        expect(s.recall).toBeLessThan(1);
        for (const code of ['What does E2 mean?', 'What should I do about E9?']) expect(s.ranks[QUESTIONS.findIndex((q) => q.q === code)]).toBe(Infinity);
      }
  });
});
