/**
 * Multi-head attention on the hand-built sentence from the attention lesson (duplicated here so
 * lessons stay independent): one head that has to look for two things at once, versus two heads
 * that each look for one, and the parameter/compute arithmetic of splitting a model into heads.
 */
export type Vec = number[];
const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0);
const matVec = (M: number[][], x: Vec) => M.map((r) => dot(r, x));

export function softmax(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Features per word: [noun, animate, pronoun, verb/adj, function word]. */
export const FEATURES = ['noun', 'animate', 'pronoun', 'verb/adj', 'function'];
export const WORDS: { w: string; x: Vec }[] = [
  { w: 'the', x: [0, 0, 0, 0, 1] },
  { w: 'animal', x: [1, 1, 0, 0, 0] },
  { w: "didn't", x: [0, 0, 0, 1, 1] },
  { w: 'cross', x: [0, 0, 0, 1, 0] },
  { w: 'the', x: [0, 0, 0, 0, 1] },
  { w: 'street', x: [1, 0, 0, 0, 0] },
  { w: 'because', x: [0, 0, 0, 0, 1] },
  { w: 'it', x: [0, 0, 1, 0, 0] },
  { w: 'was', x: [0, 0, 0, 1, 1] },
  { w: 'tired', x: [0, 0, 0, 1, 0] },
];
export const IT = 7;

export type Head = { name: string; WQ: number[][]; WK: number[][] };

/** Head 1: "who do I refer to?" — a pronoun's query asks for an animate noun. */
export const REFERENT: Head = {
  name: 'referent',
  WQ: [
    [0, 0, 3, 0, 0],
    [0, 0, 3, 0, 0],
  ],
  WK: [
    [2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
  ],
};
/** Head 2: "what is said about me?" — a pronoun's query asks for a content verb or adjective. */
export const PREDICATE: Head = {
  name: 'predicate',
  WQ: [[0, 0, 3, 0, 0]],
  WK: [[0, 0, 0, 2, -2]],
};
/** One head asked to do both jobs: its query asks for animate nouns AND for content verbs/adjectives. */
export const COMBINED: Head = {
  name: 'combined',
  WQ: [
    [0, 0, 3, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 3, 0, 0],
  ],
  WK: [
    [2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 0, 0, 2, -2],
  ],
};

/** Values pass on a word's own features. */
export const value = (x: Vec) => x.slice();

/** One head over the sentence: attention weights for every word, and every word's output. */
export function runHead(h: Head, words = WORDS) {
  const Q = words.map((t) => matVec(h.WQ, t.x));
  const K = words.map((t) => matVec(h.WK, t.x));
  const d = h.WQ.length;
  return words.map((_, i) => {
    const weights = softmax(K.map((k) => dot(Q[i], k) / Math.sqrt(d)));
    const out = FEATURES.map((_, f) => words.reduce((s, t, j) => s + weights[j] * value(t.x)[f], 0));
    return { q: Q[i], weights, out };
  });
}

/**
 * How much of each target's value reaches "it"'s output: the attention weight on the target, summed
 * over targets of that kind. With two heads, each kind arrives in its own block of the output.
 */
export function purity() {
  const one = runHead(COMBINED)[IT];
  const r = runHead(REFERENT)[IT];
  const p = runHead(PREDICATE)[IT];
  const nounShare = (w: number[]) => w[1];
  const predShare = (w: number[]) => w[3] + w[9];
  return {
    oneHead: { animal: nounShare(one.weights), predicates: predShare(one.weights), animate: one.out[1], verbAdj: one.out[3] },
    twoHeads: { animal: nounShare(r.weights), predicates: predShare(p.weights), animate: r.out[1], verbAdj: p.out[3] },
  };
}

// ---------------------------------------------------------------- cost of heads

/**
 * Parameters of one multi-head attention layer with model width d and h heads of size d/h:
 * W_Q, W_K, W_V (d × d each, all heads together) and the output mix W_O (d × d), plus biases.
 * Note that h doesn't appear: splitting into heads is free.
 */
export function attentionParams(d: number, h: number, bias = true) {
  if (d % h) throw new Error('d must be divisible by h');
  const weights = 4 * d * d;
  return { headSize: d / h, weights, biases: bias ? 4 * d : 0, total: weights + (bias ? 4 * d : 0) };
}

/** Multiply–adds to compute the scores QKᵀ and the blend (weights · V) for n tokens: h · 2 · n² · (d/h). */
export const attentionFlops = (n: number, d: number, h: number) => h * 2 * n * n * (d / h);

/**
 * One head asked to do both jobs, with a dial: `balance` 0 = only look for the referent, 1 = only
 * look for predicates; 0.5 is COMBINED. Whatever the dial, a single softmax has one 100% budget.
 */
export function combinedHead(balance: number): Head {
  const a = 2 * (1 - balance);
  const b = 2 * balance;
  return {
    name: 'combined',
    WQ: [
      [0, 0, 3 * a, 0, 0],
      [0, 0, 3 * a, 0, 0],
      [0, 0, 3 * b, 0, 0],
    ],
    WK: COMBINED.WK,
  };
}

/** Shares of "it"'s attention going to its referent and to the predicates, for one combined head. */
export function oneHeadShares(balance: number) {
  const w = runHead(combinedHead(balance))[IT].weights;
  return { referent: w[1], predicates: w[3] + w[9] };
}
