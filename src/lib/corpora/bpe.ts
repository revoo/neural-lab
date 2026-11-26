// authored-by: Claude Opus 5.5
/**
 * A small byte-pair-encoding trainer for the capstone's "characters or BPE?" comparison.
 * GPT-2 style pre-split: each word keeps its leading space ("␣king"), so merges never cross word
 * boundaries; merges are learned on word counts, which makes training fast.
 */

/** Pre-split into chunks: an optional leading space plus a run of letters/apostrophes, or a single other character. */
export const chunksOf = (s: string): string[] => s.match(/ ?[A-Za-z']+| ?[^A-Za-z' ]| /g) ?? [];

export type Bpe = { merges: [string, string][]; vocab: number };

/** Learn `n` merges from `text`. Ties are broken by the lexically smallest pair, so it's deterministic. */
export function trainBpe(text: string, n: number, baseVocab: number): Bpe {
  const counts = new Map<string, number>();
  for (const c of chunksOf(text)) counts.set(c, (counts.get(c) ?? 0) + 1);
  const words = [...counts.entries()].map(([w, k]) => ({ syms: [...w], k }));
  const merges: [string, string][] = [];
  for (let m = 0; m < n; m++) {
    const pairs = new Map<string, number>();
    for (const w of words)
      for (let i = 0; i + 1 < w.syms.length; i++) {
        const key = w.syms[i] + '\u0000' + w.syms[i + 1];
        pairs.set(key, (pairs.get(key) ?? 0) + w.k);
      }
    let best = '';
    let bestK = 0;
    for (const [key, k] of pairs) if (k > bestK || (k === bestK && key < best)) [best, bestK] = [key, k];
    if (!bestK) break;
    const [a, b] = best.split('\u0000');
    merges.push([a, b]);
    for (const w of words) {
      const out: string[] = [];
      for (let i = 0; i < w.syms.length; i++) {
        if (i + 1 < w.syms.length && w.syms[i] === a && w.syms[i + 1] === b) {
          out.push(a + b);
          i++;
        } else out.push(w.syms[i]);
      }
      w.syms = out;
    }
  }
  return { merges, vocab: baseVocab + merges.length };
}

/** Tokenise with the learned merges, applied in order. */
export function encodeBpe(bpe: Bpe, text: string): string[] {
  const out: string[] = [];
  const cache = new Map<string, string[]>();
  for (const c of chunksOf(text)) {
    let toks = cache.get(c);
    if (!toks) {
      let syms = [...c];
      for (const [a, b] of bpe.merges) {
        if (syms.length < 2) break;
        const next: string[] = [];
        for (let i = 0; i < syms.length; i++) {
          if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
            next.push(a + b);
            i++;
          } else next.push(syms[i]);
        }
        syms = next;
      }
      toks = syms;
      cache.set(c, toks);
    }
    out.push(...toks);
  }
  return out;
}
