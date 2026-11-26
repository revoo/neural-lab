// authored-by: Claude Opus 5.5
/**
 * Probes for "learning to speak" (V1, B2): what the tiny GPT's heads and embeddings do, measured.
 * Definitions are fixed here, before they were computed on any run.
 */
import { Gpt } from './gpt';
import { createRng } from '../rng';
import { CHARS, encode } from '../corpora/grimm';

/** Sequences of `L` random lower-case letters, each repeated twice (length 2L): the induction test. */
export function repeatBatch(n: number, L: number, seed = 31): Int32Array[] {
  const r = createRng(seed);
  const letters = encode('abcdefghijklmnopqrstuvwxyz');
  return Array.from({ length: n }, () => {
    const half = Array.from({ length: L }, () => letters[r.int(26)]);
    return Int32Array.from([...half, ...half]);
  });
}

export type HeadScores = {
  /** [layer][head]: mean attention from position i to i − 1. */
  prev: number[][];
  /** [layer][head]: in the second copy, mean attention from position i to i − L + 1 (the token after the earlier occurrence). */
  induction: number[][];
  /** Mean next-character loss on the first copy and on the second copy. */
  lossFirst: number;
  lossSecond: number;
};

export function headScores(model: Gpt, seqs: Int32Array[]): HeadScores {
  const { layers: Ly, heads: H, vocab: V } = model.cfg;
  const T = seqs[0].length;
  const L = T / 2;
  const prev = Array.from({ length: Ly }, () => new Array(H).fill(0));
  const ind = Array.from({ length: Ly }, () => new Array(H).fill(0));
  let l1 = 0;
  let l2 = 0;
  for (const s of seqs) {
    const targets = Int32Array.from({ length: T }, (_, t) => (t + 1 < T ? s[t + 1] : -1));
    model.forward(s, 1, T, targets);
    // per-position loss from the probabilities left in the workspace
    const att = model.attention();
    const probs = model.lastProbs();
    for (let t = 0; t + 1 < T; t++) {
      const nll = -Math.log(Math.max(probs[t * V + s[t + 1]], 1e-30));
      if (t < L - 1) l1 += nll;
      else if (t >= L) l2 += nll;
    }
    att.forEach((layer, l) =>
      layer.forEach((a, h) => {
        let p = 0;
        for (let i = 1; i < T; i++) p += a[i * T + i - 1];
        prev[l][h] += p / (T - 1);
        let q = 0;
        for (let i = L; i < T; i++) q += a[i * T + (i - L + 1)];
        ind[l][h] += q / L;
      }),
    );
  }
  const n = seqs.length;
  return {
    prev: prev.map((r) => r.map((v) => v / n)),
    induction: ind.map((r) => r.map((v) => v / n)),
    lossFirst: l1 / (n * (L - 1)),
    lossSecond: l2 / (n * (L - 1)),
  };
}

// ------------------------------------------------------------------ embeddings

export const CHAR_GROUPS = ['vowel', 'consonant', 'capital', 'digit', 'punctuation', 'space'] as const;
export type CharGroup = (typeof CHAR_GROUPS)[number];
export function charGroup(c: string): CharGroup {
  if ('aeiou'.includes(c)) return 'vowel';
  if (/[a-z]/.test(c)) return 'consonant';
  if (/[A-Z]/.test(c)) return 'capital';
  if (/[0-9]/.test(c)) return 'digit';
  if (c === ' ' || c === '\n') return 'space';
  return 'punctuation';
}
export const CHAR_GROUP_IDS = [...CHARS].map((c) => CHAR_GROUPS.indexOf(charGroup(c)));

/** The token-embedding rows of a model, one per character. */
export function embeddings(model: Gpt): Float64Array[] {
  const { vocab: V, d } = model.cfg;
  const o = model.layout.wte;
  return Array.from({ length: V }, (_, k) => Float64Array.from(model.p.subarray(o + k * d, o + (k + 1) * d)));
}

/** Leave-one-out nearest-group-average accuracy (Euclidean) of rows labelled by `groups`. */
export function looCentroidAccuracy(rows: ArrayLike<number>[], groups: readonly number[]): number {
  const G = Math.max(...groups) + 1;
  const D = rows[0].length;
  const sum = Array.from({ length: G }, () => new Float64Array(D));
  const cnt = new Array(G).fill(0);
  rows.forEach((r, i) => {
    cnt[groups[i]]++;
    for (let j = 0; j < D; j++) sum[groups[i]][j] += r[j];
  });
  let ok = 0;
  rows.forEach((r, i) => {
    let best = -1;
    let bd = Infinity;
    for (let g = 0; g < G; g++) {
      const c = cnt[g] - (g === groups[i] ? 1 : 0);
      if (!c) continue;
      let d = 0;
      for (let j = 0; j < D; j++) {
        const m = (sum[g][j] - (g === groups[i] ? r[j] : 0)) / c;
        d += (r[j] - m) ** 2;
      }
      if (d < bd) [bd, best] = [d, g];
    }
    if (best === groups[i]) ok++;
  });
  return ok / rows.length;
}

/** The q-quantile of the same accuracy with the group labels shuffled (`n` seeded shuffles). */
export function shuffledBar(rows: ArrayLike<number>[], groups: readonly number[], n = 200, q = 0.99, seed = 5): number {
  const r = createRng(seed);
  const acc: number[] = [];
  for (let k = 0; k < n; k++) {
    const g = [...groups];
    for (let i = g.length - 1; i > 0; i--) {
      const j = r.int(i + 1);
      [g[i], g[j]] = [g[j], g[i]];
    }
    acc.push(looCentroidAccuracy(rows, g));
  }
  acc.sort((a, b) => a - b);
  return acc[Math.floor(q * (n - 1))];
}

/** Top-2 principal components of row vectors (power iteration, deterministic start, sign fixed). */
export function pca2(rows: ArrayLike<number>[]): [number, number][] {
  const n = rows.length;
  const D = rows[0].length;
  const mean = new Float64Array(D);
  rows.forEach((r) => {
    for (let j = 0; j < D; j++) mean[j] += r[j] / n;
  });
  const X = rows.map((r) => Float64Array.from({ length: D }, (_, j) => r[j] - mean[j]));
  const axes: Float64Array[] = [];
  for (let c = 0; c < 2; c++) {
    let v = Float64Array.from({ length: D }, (_, j) => 1 + j / D);
    for (let it = 0; it < 300; it++) {
      const w = new Float64Array(D);
      for (const x of X) {
        let s = 0;
        for (let j = 0; j < D; j++) s += x[j] * v[j];
        for (let j = 0; j < D; j++) w[j] += s * x[j];
      }
      for (const a of axes) {
        let s = 0;
        for (let j = 0; j < D; j++) s += w[j] * a[j];
        for (let j = 0; j < D; j++) w[j] -= s * a[j];
      }
      const norm = Math.hypot(...w) || 1;
      v = w.map((x) => x / norm);
    }
    let big = 0;
    for (let j = 1; j < D; j++) if (Math.abs(v[j]) > Math.abs(v[big])) big = j;
    if (v[big] < 0) v = v.map((x) => -x);
    axes.push(v);
  }
  return X.map((x) => axes.map((a) => x.reduce((s, xj, j) => s + xj * a[j], 0)) as [number, number]);
}

/**
 * What perfectly even causal attention would score on the probe sequences (length 2L): position i
 * spreads 1/(i+1) over positions 0 … i. Returns [previous-token share, induction share].
 */
export function uniformShares(L: number): [number, number] {
  const T = 2 * L;
  let p = 0;
  for (let i = 1; i < T; i++) p += 1 / (i + 1);
  let q = 0;
  for (let i = L; i < T; i++) q += 1 / (i + 1);
  return [p / (T - 1), q / L];
}
/** A head counts as a previous-token or induction head at twice the even-attention share. */
export const HEAD_BAR_FACTOR = 2;
export const PROBE_L = 24;
