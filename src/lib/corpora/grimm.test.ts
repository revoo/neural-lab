// authored-by: Claude Opus 5.5
import text from './grimm.txt?raw';
import { describe, expect, it } from 'vitest';
import { CHARS, charFrequencies, decode, dictionary, encode, evalWindows, ngramLoss, prepareCorpus, sampleBatches, unigramKL, VOCAB, wordsOf } from './grimm';
import { createRng } from '../rng';

const c = prepareCorpus(text);

describe('the Grimm corpus', () => {
  it('has the size, vocabulary and split the lessons quote', () => {
    expect(text.length).toBe(515879);
    expect(VOCAB).toBe(68);
    expect([...new Set(text)].sort().join('')).toBe(CHARS);
    expect(c.split).toBe(468612);
    expect(c.valTales).toEqual(['KING GRISLY-BEARD', 'IRON HANS', 'CAT-SKIN', 'SNOW-WHITE AND ROSE-RED']);
    expect(text.startsWith('THE GOLDEN BIRD\nA certain king had a beautiful garden')).toBe(true);
    expect(text).not.toMatch(/Gutenberg/i);
    expect(decode(encode('The king said.'))).toBe('The king said.');
  });
  it('baselines on the held-out tales: unigram 2.985, bigram 2.286 nats per character', () => {
    expect(ngramLoss(c.train, c.val, 1)).toBeCloseTo(2.985, 3);
    expect(ngramLoss(c.train, c.val, 2)).toBeCloseTo(2.286, 3);
    expect(Math.log(VOCAB)).toBeCloseTo(4.22, 2);
  });
  it('batches are windows of the text, deterministic per seed', () => {
    const a = sampleBatches(c.train, 2, 3, 8, createRng(5));
    expect(a).toEqual(sampleBatches(c.train, 2, 3, 8, createRng(5)));
    const s = decode(a.subarray(0, 9));
    expect(text.slice(0, c.split).includes(s)).toBe(true);
    const ev = evalWindows(c.val, 4, 8);
    expect(decode(ev.subarray(0, 9))).toBe(text.slice(c.split, c.split + 9));
  });
  it('word and frequency statistics', () => {
    expect(wordsOf("The king's daughter, said 'no'.")).toEqual(["the", "king's", 'daughter', 'said', 'no']);
    const dict = dictionary(text.slice(0, c.split));
    expect(dict.has('king')).toBe(true);
    expect(dict.has('qzx')).toBe(false);
    const f = charFrequencies(c.train);
    expect(unigramKL(f, text.slice(0, 20000))).toBeLessThan(0.01);
    expect(unigramKL(f, 'zzzzzzzzzzqqqqqq')).toBeGreaterThan(1);
  });
});
