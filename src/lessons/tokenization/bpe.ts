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
export type PairCount = { pair: [string, string]; count: number };
export type Step = {
  merges: number;
  tokens: number;
  vocab: number;
  /** Corpus written as fixed-width token IDs: tokens × log₂(vocab). */
  corpusBits: number;
  /** Merge table naming two existing token IDs per merge: 2 × merges × log₂(vocab). */
  tableBits: number;
  /** Minimum description length total: corpusBits + tableBits. */
  totalBits: number;
};

/** Split the corpus into words with counts; each word starts as characters plus "_". */
export function wordsOf(text: string): Word[] {
  const counts = new Map<string, number>();
  for (const w of text.toLowerCase().split(/\s+/).filter(Boolean)) counts.set(w, (counts.get(w) ?? 0) + 1);
  return [...counts.entries()].map(([w, count]) => ({ symbols: [...w, '_'], count }));
}

export function initialSymbols(words: Word[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const w of words)
    for (const s of w.symbols)
      if (!seen.has(s)) {
        seen.add(s);
        out.push(s);
      }
  return out;
}

export function countPairs(words: Word[]) {
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

/** Most frequent adjacent pairs in the current word list (ties broken by first appearance). */
export function topPairs(words: Word[], limit = 12): PairCount[] {
  return [...countPairs(words).values()]
    .sort((a, b) => b.count - a.count || a.first - b.first)
    .slice(0, limit)
    .map(({ pair, count }) => ({ pair, count }));
}

export function applyMerge(symbols: string[], a: string, b: string) {
  const out: string[] = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(symbols[i]);
  }
  return out;
}

export function applyMergeToWords(words: Word[], a: string, b: string): Word[] {
  return words.map((w) => ({ ...w, symbols: applyMerge(w.symbols, a, b) }));
}

export const totalTokens = (words: Word[]) => words.reduce((s, w) => s + w.symbols.length * w.count, 0);

/** Minimum description length cost in bits: corpus as token IDs + merge table (2 IDs per merge). */
export function mdlStep(words: Word[], merges: number, initialVocabSize: number): Step {
  const tokens = totalTokens(words);
  const vocab = initialVocabSize + merges;
  const logV = Math.log2(vocab);
  const corpusBits = tokens * logV;
  const tableBits = 2 * merges * logV;
  return { merges, tokens, vocab, corpusBits, tableBits, totalBits: corpusBits + tableBits };
}

/**
 * Learn up to `maxMerges` merges. Each step merges the most frequent adjacent pair (ties: the pair
 * seen first). Records the corpus's token count, vocabulary size, and minimum description length
 * cost (corpus IDs + merge table) after every merge.
 */
export function train(text = CORPUS, maxMerges = 200) {
  let words = wordsOf(text);
  const baseVocab = initialSymbols(words);
  const initialVocabSize = baseVocab.length;
  const merges: Merge[] = [];
  const steps: Step[] = [mdlStep(words, 0, initialVocabSize)];
  const snapshots: Word[][] = [words];
  for (let m = 1; m <= maxMerges; m++) {
    const pairs = [...countPairs(words).values()];
    if (!pairs.length) break;
    const best = pairs.reduce((b, p) => (p.count > b.count || (p.count === b.count && p.first < b.first) ? p : b));
    if (best.count < 2) break; // merging a pair seen once teaches nothing
    const [a, b] = best.pair;
    words = applyMergeToWords(words, a, b);
    merges.push({ pair: [a, b], merged: a + b, count: best.count });
    steps.push(mdlStep(words, m, initialVocabSize));
    snapshots.push(words);
  }
  return { merges, steps, snapshots, baseVocab, initialVocabSize };
}

/** Build the symbol → integer ID table after `k` merges. */
export function vocabIds(baseVocab: string[], merges: Merge[], k = merges.length): Map<string, number> {
  const map = new Map<string, number>();
  baseVocab.forEach((sym, i) => map.set(sym, i));
  merges.slice(0, k).forEach((m, i) => {
    if (!map.has(m.merged)) map.set(m.merged, baseVocab.length + i);
  });
  return map;
}

/** Tokenise one word with the first `k` merges, replayed in the order they were learned. */
export function encodeWord(word: string, merges: Merge[], k = merges.length): string[] {
  let s = [...word.toLowerCase(), '_'];
  for (const m of merges.slice(0, k)) s = applyMerge(s, m.pair[0], m.pair[1]);
  return s;
}

export type AppliedMerge = { rule: number; pair: [string, string]; merged: string; result: string[] };

/** Tokenise one word and record every merge rule that actually fired along the way. */
export function encodeWordTrace(word: string, merges: Merge[], k = merges.length): { tokens: string[]; applied: AppliedMerge[] } {
  let s = [...word.toLowerCase(), '_'];
  const applied: AppliedMerge[] = [];
  for (let i = 0; i < Math.min(k, merges.length); i++) {
    const m = merges[i];
    const next = applyMerge(s, m.pair[0], m.pair[1]);
    if (next.length !== s.length) {
      s = next;
      applied.push({ rule: i + 1, pair: m.pair, merged: m.merged, result: s });
    }
  }
  return { tokens: s, applied };
}

export const encodeText = (text: string, merges: Merge[], k = merges.length) =>
  text
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => encodeWord(w, merges, k));

/** The merge count with the fewest total MDL bits (corpus IDs + merge table), computed exhaustively over every step. */
export function bestMdl(steps: Step[]): Step {
  return steps.reduce((b, s) => (s.totalBits < b.totalBits ? s : b));
}

/** Same short sentence across six scripts, to measure raw UTF-8 byte counts before any merges. */
export const LANGUAGE_SAMPLES = [
  { lang: 'English', text: 'The cat sat on the warm mat.' },
  { lang: 'Spanish', text: 'El gato se sentó en la alfombra cálida.' },
  { lang: 'Greek', text: 'Η γάτα κάθισε στο ζεστό χαλάκι.' },
  { lang: 'Japanese', text: '猫は暖かいマットの上に座った。' },
  { lang: 'Hindi', text: 'बिल्ली गर्म चटाई पर बैठ गई।' },
  { lang: 'Emoji', text: '🐱🪑🔥🧶' },
] as const;

const utf8Encoder = new TextEncoder();

export function utf8Stats(text: string) {
  const chars = [...text].length;
  const bytes = [...utf8Encoder.encode(text)];
  return { chars, byteCount: bytes.length, bytes, bytesPerChar: chars > 0 ? bytes.length / chars : 0 };
}
