/**
 * The transformer block, small enough to print: a pre-norm block (LayerNorm → attention → add,
 * LayerNorm → feed-forward → add) with a causal mask, on 4-number vectors; the depth experiment
 * behind residual connections; and GPT parameter counting.
 */
import { createRng } from '../../lib/rng';

export type Vec = number[];
export type Mat = number[][];
const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0);
const matVec = (M: Mat, x: Vec) => M.map((r) => dot(r, x));
const addV = (a: Vec, b: Vec) => a.map((v, i) => v + b[i]);
export const norm = (v: Vec) => Math.sqrt(dot(v, v));

export function softmax(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Layer norm (gain 1, shift 0): subtract the vector's own mean, divide by its own spread. */
export function layerNorm(x: Vec, eps = 1e-5): Vec {
  const m = x.reduce((a, b) => a + b, 0) / x.length;
  const v = x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length;
  return x.map((t) => (t - m) / Math.sqrt(v + eps));
}

// ---------------------------------------------------------------- a tiny block, step by step

export const D = 4;
export const HIDDEN = 16;
const r = createRng(77);
const rnd = (n: number, m: number, s: number) => Array.from({ length: n }, () => Array.from({ length: m }, () => r.normal(0, s)));
export const W = {
  q: rnd(D, D, 0.8),
  k: rnd(D, D, 0.8),
  v: rnd(D, D, 0.8),
  o: rnd(D, D, 0.5),
  up: rnd(HIDDEN, D, 0.6),
  down: rnd(D, HIDDEN, 0.3),
};
export const TOKENS = ['the', 'cat', 'sat'];
export const EMB: Vec[] = rnd(3, D, 1);

export type Stage = { name: string; vecs: Vec[]; note: string; weights?: number[][] };

/** Run the block on all tokens, recording every intermediate stage. */
export function runBlock(X: Vec[] = EMB, causal = true): Stage[] {
  const n = X.length;
  const n1 = X.map((x) => layerNorm(x));
  const Q = n1.map((x) => matVec(W.q, x));
  const K = n1.map((x) => matVec(W.k, x));
  const V = n1.map((x) => matVec(W.v, x));
  const weights = Q.map((q, i) => softmax(K.map((k, j) => (causal && j > i ? -Infinity : dot(q, k) / Math.sqrt(D)))));
  const mixed = weights.map((w) => V[0].map((_, c) => w.reduce((s, wi, j) => s + wi * V[j][c], 0)));
  const attn = mixed.map((m) => matVec(W.o, m));
  const h = X.map((x, i) => addV(x, attn[i]));
  const n2 = h.map((x) => layerNorm(x));
  const up = n2.map((x) => matVec(W.up, x).map((v) => Math.max(0, v)));
  const ffn = up.map((u) => matVec(W.down, u));
  const out = h.map((x, i) => addV(x, ffn[i]));
  void n;
  return [
    { name: 'x', vecs: X, note: 'the token vectors entering the block (word + position)' },
    { name: 'LN(x)', vecs: n1, note: 'layer norm: each vector shifted to mean 0 and scaled to spread 1' },
    { name: 'attention', vecs: attn, note: 'what attention gathered from the other tokens', weights },
    { name: 'h = x + attention', vecs: h, note: 'residual: the gathered information is ADDED to the original' },
    { name: 'LN(h)', vecs: n2, note: 'normalise again before the feed-forward network' },
    { name: 'FFN(LN(h))', vecs: ffn, note: 'feed-forward: each token on its own, 4 → 16 (ReLU) → 4' },
    { name: 'out = h + FFN', vecs: out, note: 'residual again: the block’s output, ready for the next block' },
  ];
}

// ---------------------------------------------------------------- why residuals: the depth experiment

/**
 * A stack of `depth` random sublayers f(x) = tanh(W x) on vectors of length `width`, with or
 * without residual connections (x ← x + f(x)) and with or without layer norm before each sublayer.
 * Returns the size of the gradient that reaches the input (‖∂ output/∂ input · u‖ for a random unit
 * direction u, estimated by finite differences), and the size of the output — averaged over seeds.
 */
export function depthStudy(depth: number, residual: boolean, normed: boolean, width = 32, seeds = 8, gain = 1) {
  let grad = 0;
  let size = 0;
  for (let s = 1; s <= seeds; s++) {
    const rr = createRng(s * 131 + depth);
    const Ws = Array.from({ length: depth }, () => Array.from({ length: width }, () => Array.from({ length: width }, () => rr.normal(0, gain / Math.sqrt(width)))));
    const x0 = Array.from({ length: width }, () => rr.normal());
    const u = Array.from({ length: width }, () => rr.normal());
    const un = norm(u);
    const forward = (x: Vec) => {
      let h = x;
      for (const Wl of Ws) {
        const inp = normed ? layerNorm(h) : h;
        const f = matVec(Wl, inp).map(Math.tanh);
        h = residual ? addV(h, f) : f;
      }
      return h;
    };
    const eps = 1e-5;
    const a = forward(x0.map((v, i) => v + (eps * u[i]) / un));
    const b = forward(x0.map((v, i) => v - (eps * u[i]) / un));
    grad += norm(a.map((v, i) => (v - b[i]) / (2 * eps)));
    size += norm(forward(x0)) / Math.sqrt(width);
  }
  return { depth, grad: grad / seeds, size: size / seeds };
}

// ---------------------------------------------------------------- counting a GPT's parameters

export type GptShape = { d: number; layers: number; vocab: number; context: number; ffnMult?: number };

/** Parameters of a GPT-2-style model (learned positions, tied output embedding, biases everywhere). */
export function gptParams({ d, layers, vocab, context, ffnMult = 4 }: GptShape) {
  const attn = 4 * d * d + 4 * d;
  const ffn = 2 * ffnMult * d * d + ffnMult * d + d;
  const norms = 2 * 2 * d;
  const block = attn + ffn + norms;
  const embed = vocab * d;
  const pos = context * d;
  const finalNorm = 2 * d;
  return { attn, ffn, norms, block, blocks: block * layers, embed, pos, finalNorm, total: block * layers + embed + pos + finalNorm };
}

export const GPT2 = {
  small: { d: 768, layers: 12, vocab: 50257, context: 1024 },
  medium: { d: 1024, layers: 24, vocab: 50257, context: 1024 },
  large: { d: 1280, layers: 36, vocab: 50257, context: 1024 },
  xl: { d: 1600, layers: 48, vocab: 50257, context: 1024 },
};
