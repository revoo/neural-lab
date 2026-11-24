/**
 * Byte-pair encoding, small enough to watch: learn merges from a corpus, then tokenise new text by
 * replaying them. Words are split on whitespace and each ends with "_", the end-of-word marker, so
 * "low_" (a whole word) and "low" (the start of "lower") can become different tokens.
 */

export const CORPUS = `the cat sat on the mat and the cat saw the rat
a fast cat chased a slow rat across the warm mat
the rats were faster than the cats at night
cats sleep in the sun and rats sleep in the dark
the slowest rat sat still while the fastest cat watched
she said the cat was watching the rats again
watching and waiting the cats sat on the warm mats
the oldest cat was the slowest and the youngest was the fastest
new cats and old rats shared the warmest corner of the barn
strawberries and raspberries grew behind the barn in the warm sun`;

export type Word = { symbols: string[]; count: number };
export type Merge = { pair: [string, string]; merged: string; count: number };
export type Step = { merges: number; tokens: number; vocab: number; bits: number };

/** Split the corpus into words with counts; each word starts as characters plus "_". */
export function wordsOf(text: string): Word[] {
  const counts = new Map<string, number>();
  for (const w of text.toLowerCase().split(/\s+/).filter(Boolean)) counts.set(w, (counts.get(w) ?? 0) + 1);
  return [...counts.entries()].map(([w, count]) => ({ symbols: [...w, '_'], count }));
}

const initialVocab = (words: Word[]) => new Set(words.flatMap((w) => w.symbols));

function countPairs(words: Word[]) {
  const pairs = new Map<string, { pair: [string, string]; count: number; first: number }>();
  let order = 0;
  for (const w of words)
    for (let i = 0; i < w.symbols.length - 1; i++) {
      const key = w.symbols[i] + '\u0000' + w.symbols[i + 1];
      const e = pairs.get(key);
      if (e) e.count += w.count;
      else pairs.set(key, { pair: [w.symbols[i], w.symbols[i + 1]], count: w.count, first: order++ });
    }
  return pairs;
}

function applyMerge(symbols: string[], a: string, b: string) {
  const out: string[] = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(symbols[i]);
  }
  return out;
}

const totalTokens = (words: Word[]) => words.reduce((s, w) => s + w.symbols.length * w.count, 0);

/**
 * Learn up to `maxMerges` merges. Each step merges the most frequent adjacent pair (ties: the pair
 * seen first). Records the corpus's token count, the vocabulary size, and the bits needed to write
 * the corpus as token ids, tokens × log₂(vocabulary), after every merge.
 */
export function train(text = CORPUS, maxMerges = 200) {
  let words = wordsOf(text);
  const vocab = initialVocab(words);
  const merges: Merge[] = [];
  const step = (m: number): Step => {
    const tokens = totalTokens(words);
    return { merges: m, tokens, vocab: vocab.size, bits: tokens * Math.log2(vocab.size) };
  };
  const steps: Step[] = [step(0)];
  const snapshots: Word[][] = [words];
  for (let m = 1; m <= maxMerges; m++) {
    const pairs = [...countPairs(words).values()];
    if (!pairs.length) break;
    const best = pairs.reduce((b, p) => (p.count > b.count || (p.count === b.count && p.first < b.first) ? p : b));
    if (best.count < 2) break; // merging a pair seen once teaches nothing
    const [a, b] = best.pair;
    words = words.map((w) => ({ ...w, symbols: applyMerge(w.symbols, a, b) }));
    vocab.add(a + b);
    merges.push({ pair: [a, b], merged: a + b, count: best.count });
    steps.push(step(m));
    snapshots.push(words);
  }
  return { merges, steps, snapshots, initialVocabSize: initialVocab(wordsOf(text)).size };
}

/** Tokenise one word with the first `k` merges, replayed in the order they were learned. */
export function encodeWord(word: string, merges: Merge[], k = merges.length): string[] {
  let s = [...word.toLowerCase(), '_'];
  for (const m of merges.slice(0, k)) s = applyMerge(s, m.pair[0], m.pair[1]);
  return s;
}

export const encodeText = (text: string, merges: Merge[], k = merges.length) =>
  text
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => encodeWord(w, merges, k));

/** The number of merges with the fewest total bits (tokens × log₂ vocabulary): computed over every step. */
export function bestBits(steps: Step[]) {
  return steps.reduce((b, s) => (s.bits < b.bits ? s : b));
}
