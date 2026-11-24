/** Probability helpers: softmax, entropy, cross-entropy. All logs are natural unless stated. */

/** Numerically stable softmax with optional temperature. */
export function softmax(logits: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  const scaled = logits.map((z) => z / t);
  const m = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

export function logSumExp(xs: number[]): number {
  const m = Math.max(...xs);
  return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0));
}

/** Entropy of a distribution, in the given log base (default 2 → bits). */
export function entropy(p: number[], base = 2): number {
  const lb = Math.log(base);
  return -p.reduce((s, pi) => (pi > 0 ? s + (pi * Math.log(pi)) / lb : s), 0);
}

/** Cross-entropy H(p, q) = -Σ p log q. */
export function crossEntropy(p: number[], q: number[], base = Math.E): number {
  const lb = Math.log(base);
  return -p.reduce((s, pi, i) => (pi > 0 ? s + (pi * Math.log(Math.max(q[i], 1e-300))) / lb : s), 0);
}

/** Surprise (information content) of an event with probability p. */
export const surprise = (p: number, base = 2): number => -Math.log(p) / Math.log(base);
