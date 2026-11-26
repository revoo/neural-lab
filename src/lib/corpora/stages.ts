// authored-by: Claude Opus 5.5
/**
 * Measured stages of "learning to speak" (V1, A3/B2). Each statistic is computed on the model's
 * samples, and each stage's threshold is calibrated on real passages of the same length from the
 * training tales, so "it has reached stage X" means "on this statistic, its text is as close to the
 * tales as a genuine passage usually is".
 */
import { charFrequencies, encode, unigramKL, wordsOf } from './grimm';

export type SampleStats = {
  /** Nats between the corpus's letter frequencies and the sample's. */
  kl: number;
  /** Mean length of the words (runs of letters). */
  meanWordLen: number;
  /** Share of words that occur somewhere in the training tales. */
  realShare: number;
  /** Longest run of consecutive real words. */
  longestRun: number;
  /** Distinct real words of three or more letters. */
  distinctReal: number;
};

export function sampleStats(text: string, dict: Set<string>, freq: Float64Array): SampleStats {
  const words = wordsOf(text);
  let run = 0;
  let longest = 0;
  let real = 0;
  const distinct = new Set<string>();
  for (const w of words) {
    if (dict.has(w)) {
      real++;
      run++;
      longest = Math.max(longest, run);
      if (w.length >= 3) distinct.add(w);
    } else run = 0;
  }
  return {
    kl: unigramKL(freq, text),
    meanWordLen: words.length ? words.reduce((s, w) => s + w.length, 0) / words.length : 0,
    realShare: words.length ? real / words.length : 0,
    longestRun: longest,
    distinctReal: distinct.size,
  };
}

/** Quantile q of xs (linear interpolation). */
export const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  const p = (s.length - 1) * q;
  const lo = Math.floor(p);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (p - lo);
};

export type Calibration = {
  /** Real passages of `len` characters: 95th percentile of their letter-mix KL. */
  klP95: number;
  /** Corpus mean word length, and the 95th percentile of a passage's distance from it. */
  wordLen: number;
  wordLenP95: number;
  /** 5th percentile of the longest real-word run in a real passage (real text is all real words). */
  runP05: number;
};

/** Statistics of `n` evenly spaced real passages of `len` characters from the training text. */
export function calibrate(train: string, len: number, n: number, dict: Set<string>): Calibration {
  const freq = charFrequencies(encode(train));
  const all = wordsOf(train);
  const wordLen = all.reduce((s, w) => s + w.length, 0) / all.length;
  const st = Array.from({ length: n }, (_, i) => {
    const o = Math.floor((i * (train.length - len)) / (n - 1));
    return sampleStats(train.slice(o, o + len), dict, freq);
  });
  return {
    klP95: quantile(st.map((s) => s.kl), 0.95),
    wordLen,
    wordLenP95: quantile(st.map((s) => Math.abs(s.meanWordLen - wordLen)), 0.95),
    runP05: quantile(st.map((s) => s.longestRun), 0.05),
  };
}

export type StageId = 'letters' | 'spacing' | 'half-real' | 'most-real' | 'phrase';
export type Stage = { id: StageId; label: string; test: (s: SampleStats, c: Calibration) => boolean };

/** The stages, in the order a learner would expect them; each is a stated, checkable criterion. */
export const STAGES: Stage[] = [
  { id: 'letters', label: 'the right mix of letters', test: (s, c) => s.kl <= c.klP95 },
  { id: 'spacing', label: 'words of the right length', test: (s, c) => Math.abs(s.meanWordLen - c.wordLen) <= c.wordLenP95 },
  { id: 'half-real', label: 'half its words are real', test: (s) => s.realShare >= 0.5 },
  { id: 'most-real', label: 'four words in five are real', test: (s) => s.realShare >= 0.8 },
  { id: 'phrase', label: 'six real words in a row', test: (s) => s.longestRun >= 6 },
];

/**
 * For each stage, the first checkpoint step at which it holds and keeps holding at every later
 * checkpoint ("for good"), or null if it never settles.
 */
export function stageSteps(history: readonly { step: number; stats: SampleStats }[], c: Calibration): Record<StageId, number | null> {
  const out = {} as Record<StageId, number | null>;
  for (const st of STAGES) {
    let from: number | null = null;
    for (const h of history) {
      if (st.test(h.stats, c)) from ??= h.step;
      else from = null;
    }
    out[st.id] = from;
  }
  return out;
}
