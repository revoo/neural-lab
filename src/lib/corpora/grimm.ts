// authored-by: Claude Opus 5.5
/**
 * The tiny GPT's corpus: Grimms' Fairy Tales (Project Gutenberg eBook #2591, the Edgar Taylor and
 * Marian Edwardes translation, public domain), cleaned by `buildGrimm.run.ts` into `grimm.txt`.
 *
 * Pure functions over the text, so Node scripts, tests and widgets share them. Widgets load the
 * text lazily with `loadGrimm()` (a separate chunk, ~175 KB gzipped); Node reads the file.
 */
import { createRng } from '../rng';

/** The 68 characters of the cleaned corpus, in code-point order: token id = index. */
export const CHARS = '\n !"\'(),-.123:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
export const VOCAB = CHARS.length;
const INDEX = new Map([...CHARS].map((c, i) => [c, i]));

export const encode = (s: string): Int32Array => Int32Array.from([...s], (c) => {
  const i = INDEX.get(c);
  if (i === undefined) throw new Error(`character ${JSON.stringify(c)} is not in the vocabulary`);
  return i;
});
export const decode = (ids: ArrayLike<number>): string => Array.from(ids, (i) => CHARS[i]).join('');

export type Corpus = {
  text: string;
  ids: Int32Array;
  /** Index of the first validation character (the start of a tale). */
  split: number;
  train: Int32Array;
  val: Int32Array;
  /** Titles of the tales held out for validation. */
  valTales: string[];
};

/** Headings are whole lines in capitals ("THE GOLDEN BIRD"). */
export function taleStarts(text: string): { at: number; title: string }[] {
  const out: { at: number; title: string }[] = [];
  let at = 0;
  for (const line of text.split('\n')) {
    if (line.length > 2 && line === line.toUpperCase() && /[A-Z]/.test(line) && !/[a-z]/.test(line)) out.push({ at, title: line });
    at += line.length + 1;
  }
  return out;
}

/** Hold out the tales from the heading nearest 90% of the text onwards. */
export function prepareCorpus(text: string): Corpus {
  const ids = encode(text);
  const starts = taleStarts(text);
  const target = 0.9 * text.length;
  let best = starts[0];
  for (const s of starts) if (Math.abs(s.at - target) < Math.abs(best.at - target)) best = s;
  const split = best.at;
  return {
    text,
    ids,
    split,
    train: ids.subarray(0, split),
    val: ids.subarray(split),
    valTales: starts.filter((s) => s.at >= split).map((s) => s.title),
  };
}

/** Lazily load the corpus in the browser (Vite turns the ?raw import into its own chunk). */
export async function loadGrimm(): Promise<Corpus> {
  const mod = await import('./grimm.txt?raw');
  return prepareCorpus(mod.default);
}

/**
 * K batches of B windows of T + 1 consecutive tokens, at uniformly random offsets in `ids`, laid
 * out [K × B × (T+1)] (the layout the trainers take). The RNG is advanced in that order.
 */
export function sampleBatches(ids: Int32Array, K: number, B: number, T: number, rng: ReturnType<typeof createRng>): Int32Array {
  const out = new Int32Array(K * B * (T + 1));
  const span = ids.length - (T + 1);
  for (let r = 0; r < K * B; r++) {
    const o = rng.int(span + 1);
    out.set(ids.subarray(o, o + T + 1), r * (T + 1));
  }
  return out;
}

/** A fixed evaluation set: n windows of T + 1 tokens at evenly spaced offsets (no randomness). */
export function evalWindows(ids: Int32Array, n: number, T: number): Int32Array {
  const out = new Int32Array(n * (T + 1));
  const span = ids.length - (T + 1);
  for (let r = 0; r < n; r++) {
    const o = Math.floor((r * span) / Math.max(1, n - 1));
    out.set(ids.subarray(o, o + T + 1), r * (T + 1));
  }
  return out;
}

// ------------------------------------------------------------------ baselines

/**
 * Cross-entropy (nats per character) on `test` of an n-gram character model counted on `train`,
 * with add-k smoothing. n = 1: unigram (letter frequencies); n = 2: bigram (which letter follows which).
 */
export function ngramLoss(train: Int32Array, test: Int32Array, n: 1 | 2, k = 0.5): number {
  const V = VOCAB;
  if (n === 1) {
    const c = new Float64Array(V).fill(k);
    for (const t of train) c[t]++;
    const tot = c.reduce((a, b) => a + b, 0);
    let s = 0;
    for (const t of test) s -= Math.log(c[t] / tot);
    return s / test.length;
  }
  const c = new Float64Array(V * V).fill(k);
  const row = new Float64Array(V).fill(k * V);
  for (let i = 0; i + 1 < train.length; i++) {
    c[train[i] * V + train[i + 1]]++;
    row[train[i]]++;
  }
  let s = 0;
  for (let i = 0; i + 1 < test.length; i++) s -= Math.log(c[test[i] * V + test[i + 1]] / row[test[i]]);
  return s / (test.length - 1);
}

// ------------------------------------------------------------------ what a sample contains

/** Words as the lesson counts them: runs of letters (an apostrophe inside a word is kept), lower-cased. */
export const wordsOf = (s: string): string[] => (s.toLowerCase().match(/[a-z]+(?:'[a-z]+)*/g) ?? []);

/** Every word that occurs in the training text: the "is it a real word?" dictionary. */
export function dictionary(train: string): Set<string> {
  return new Set(wordsOf(train));
}

/** KL divergence (nats) from the corpus's character frequencies to a sample's (add-½ smoothing on the sample). */
export function unigramKL(corpusFreq: Float64Array, sample: string): number {
  const V = VOCAB;
  const c = new Float64Array(V).fill(0.5);
  for (const ch of sample) {
    const i = INDEX.get(ch);
    if (i !== undefined) c[i]++;
  }
  const tot = c.reduce((a, b) => a + b, 0);
  let kl = 0;
  for (let i = 0; i < V; i++) if (corpusFreq[i] > 0) kl += corpusFreq[i] * Math.log(corpusFreq[i] / (c[i] / tot));
  return kl;
}

export function charFrequencies(ids: Int32Array): Float64Array {
  const f = new Float64Array(VOCAB);
  for (const t of ids) f[t]++;
  for (let i = 0; i < VOCAB; i++) f[i] /= ids.length;
  return f;
}
