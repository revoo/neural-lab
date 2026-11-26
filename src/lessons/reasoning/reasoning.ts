// authored-by: Claude Opus 5.5
/**
 * Test-time compute on a transparent stand-in for a language model.
 *
 * No language model runs in the browser, so the lesson uses a stand-in with the same structure as the
 * real thing and says so. A "problem" needs d reasoning steps. A model tier gets each step it writes
 * out right with probability r (its per-step reliability); steps it skips (because its thinking
 * budget ran out) it must do "in its head", where it is far less reliable (r⁴ per step). So one
 * sample is correct with probability
 *     p = r^w · (r⁴)^(d − w),   w = min(d, ⌊thinking tokens / 40⌋).
 * When a sample is wrong, a share of the wrong answers land on the problem's one tempting "trap"
 * answer; the rest are all different from each other.
 *
 * Given p, everything else is exact (no sampling): pass@N, majority vote (by enumerating the
 * trinomial counts of correct, trap and other answers), and best-of-N under a verifier, in closed form.
 */

export type Tier = 'small' | 'medium' | 'large';
export const TIERS: Record<Tier, { label: string; params: number; r: number }> = {
  small: { label: 'Small (1B)', params: 1e9, r: 0.8 },
  medium: { label: 'Medium (8B)', params: 8e9, r: 0.9 },
  large: { label: 'Large (70B)', params: 70e9, r: 0.96 },
};
export const TOKENS_PER_STEP = 40;
export const ANSWER_TOKENS = 20;
/** The noisy reward model is itself an 8B model that reads every sample. */
export const RM_PARAMS = 8e9;

export type Problem = { id: number; steps: number; trap: number };

/** The benchmark: 50 problems, 1 to 10 steps, with trap shares from 0 to 0.8. */
export const BENCH: Problem[] = Array.from({ length: 50 }, (_, i) => ({ id: i, steps: 1 + (i % 10), trap: [0, 0.2, 0.4, 0.6, 0.8][Math.floor(i / 10)] }));

export function pSingle(tier: Tier, steps: number, thinking: number) {
  const r = TIERS[tier].r;
  const w = Math.min(steps, Math.floor(thinking / TOKENS_PER_STEP));
  return r ** w * r ** (4 * (steps - w));
}

export const passAtN = (p: number, n: number) => 1 - (1 - p) ** n;

/** log-factorials for the trinomial */
const LF: number[] = [0];
for (let i = 1; i <= 512; i++) LF[i] = LF[i - 1] + Math.log(i);

/**
 * Majority vote over n samples: probability that the most common answer is the correct one (ties
 * broken uniformly at random). Answers are: correct (p), the trap (t), or one of infinitely many
 * other wrong answers (o), so "other" answers never repeat.
 */
export function majority(p: number, trap: number, n: number) {
  const t = (1 - p) * trap;
  const o = 1 - p - t;
  let acc = 0;
  for (let c = 0; c <= n; c++)
    for (let k = 0; k + c <= n; k++) {
      const r = n - c - k;
      const lp = LF[n] - LF[c] - LF[k] - LF[r] + (c ? c * Math.log(p) : 0) + (k ? k * Math.log(t) : 0) + (r ? r * Math.log(o) : 0);
      if (!Number.isFinite(lp)) continue;
      const pr = Math.exp(lp);
      const m = Math.max(c, k, r > 0 ? 1 : 0);
      if (c !== m) continue;
      const tied = m === 1 ? (c === 1 ? 1 : 0) + (k === 1 ? 1 : 0) + r : 1 + (k === m ? 1 : 0);
      acc += pr / tied;
    }
  return acc;
}

export type Verifier = 'none' | 'noisy' | 'exact';
export const VERIFIERS: Record<Verifier, string> = { none: 'None', noisy: 'Noisy reward model (85%)', exact: 'Exact checker (unit tests)' };
/** The noisy reward model approves a correct answer 85% of the time and rejects an ordinary wrong one 85% of the time… */
export const RM_ACC = 0.85;
/** …but a small share of wrong answers are polished enough to fool it completely: it scores them above everything. */
export const RM_BLIND_SPOT = 0.02;

/**
 * Best-of-n: draw n samples, keep the one the verifier scores highest (ties at random).
 * - none: the pick is a random sample: accuracy p.
 * - exact: any correct sample wins: pass@n.
 * - noisy: scores are 2 for a blind-spot wrong answer, 1 for an approved answer, 0 for a rejected one.
 *   Because samples are independent, the chosen sample is distributed like one sample conditioned on
 *   the top score present, which gives a closed form.
 */
export function bestOfN(p: number, n: number, v: Verifier) {
  if (v === 'none') return p;
  if (v === 'exact') return passAtN(p, n);
  const e = (1 - p) * RM_BLIND_SPOT;
  const c1 = RM_ACC * p;
  const w1 = (1 - RM_ACC) * (1 - p - e);
  const c0 = (1 - RM_ACC) * p;
  const w0 = RM_ACC * (1 - p - e);
  const noE = (1 - e) ** n;
  const allZero = ((c0 + w0) / (1 - e)) ** n;
  return noE * ((1 - allZero) * (c1 / (c1 + w1)) + allZero * (c0 / (c0 + w0 || 1)));
}

export type Strategy = 'single' | 'majority' | 'best';
export const STRATEGIES: Record<Strategy, string> = { single: '1-shot', majority: 'Majority vote', best: 'Best-of-N' };

export type Config = { tier: Tier; strategy: Strategy; n: number; verifier: Verifier; thinking: number };

export function solveRate(c: Config, prob: Problem) {
  const p = pSingle(c.tier, prob.steps, c.thinking);
  if (c.strategy === 'single') return p;
  if (c.strategy === 'majority') return majority(p, prob.trap, c.n);
  return bestOfN(p, c.n, c.verifier);
}

export const samples = (c: Config) => (c.strategy === 'single' ? 1 : c.n);
export const tokensPerSample = (thinking: number) => thinking + ANSWER_TOKENS;

/** Total compute per problem: 2 FLOPs per weight per generated token, plus the reward model reading each sample. */
export function flops(c: Config) {
  const toks = samples(c) * tokensPerSample(c.thinking);
  const gen = 2 * TIERS[c.tier].params * toks;
  const rm = c.strategy === 'best' && c.verifier === 'noisy' ? 2 * RM_PARAMS * toks : 0;
  return gen + rm;
}

export function benchAccuracy(c: Config, problems: Problem[] = BENCH) {
  return problems.reduce((s, q) => s + solveRate(c, q), 0) / problems.length;
}

// ------------------------------------------------------------------ GRPO on the stand-in

/**
 * A policy that decides how many reasoning steps to write before answering (0 … 12), one softmax per
 * problem difficulty (1 … 10 steps). It starts out preferring to answer at once, like a model that
 * hasn't been taught to think out loud. Writing a step costs nothing but can slip: P(correct) =
 * r^L · (r⁴)^(d − L) if L < d, and r^L if L ≥ d (extra steps are extra chances to slip).
 *
 * GRPO: for each problem, sample a group of G lengths, score each 1 or 0 (a verifiable reward),
 * advantage = (reward − group mean) / group standard deviation, and step the logits along
 * Σ advantage · ∇ log π(L). No value network: the group is its own baseline.
 */
export const MAX_LEN = 12;
export type Policy = number[][]; // [difficulty − 1][L]

export function initPolicy(): Policy {
  return Array.from({ length: 10 }, () => Array.from({ length: MAX_LEN + 1 }, (_, L) => -0.5 * L));
}

export const softmaxRow = (row: number[]) => {
  const m = Math.max(...row);
  const e = row.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

export function pCorrectLen(r: number, d: number, L: number) {
  return L >= d ? r ** L : r ** L * r ** (4 * (d - L));
}

/** d log π(L) / d logits = onehot(L) − π. */
export function gradLogPi(row: number[], L: number) {
  const pi = softmaxRow(row);
  return pi.map((q, j) => (j === L ? 1 : 0) - q);
}

export type GrpoStats = { step: number; acc: number; meanLen: number; lenByDiff: number[] };

/** Expected accuracy and length of the current policy over difficulties 1–10 (exact, no sampling). */
export function policyStats(pol: Policy, r: number, step = 0): GrpoStats {
  let acc = 0;
  let len = 0;
  const lenByDiff: number[] = [];
  pol.forEach((row, i) => {
    const pi = softmaxRow(row);
    let a = 0;
    let l = 0;
    pi.forEach((q, L) => {
      a += q * pCorrectLen(r, i + 1, L);
      l += q * L;
    });
    acc += a / 10;
    len += l / 10;
    lenByDiff.push(l);
  });
  return { step, acc, meanLen: len, lenByDiff };
}

export function grpoStep(pol: Policy, r: number, rng: () => number, group = 8, lr = 0.5) {
  for (let d = 1; d <= 10; d++) {
    const row = pol[d - 1];
    const pi = softmaxRow(row);
    const Ls: number[] = [];
    const rew: number[] = [];
    for (let g = 0; g < group; g++) {
      let u = rng();
      let L = 0;
      while (L < MAX_LEN && u > pi[L]) u -= pi[L++];
      Ls.push(L);
      rew.push(rng() < pCorrectLen(r, d, L) ? 1 : 0);
    }
    const mean = rew.reduce((a, b) => a + b, 0) / group;
    const sd = Math.sqrt(rew.reduce((a, b) => a + (b - mean) ** 2, 0) / group);
    if (sd === 0) continue; // every answer scored the same: no signal in this group
    const step = new Array(row.length).fill(0);
    Ls.forEach((L, g) => {
      const adv = (rew[g] - mean) / sd;
      gradLogPi(row, L).forEach((v, j) => (step[j] += (adv * v) / group));
    });
    step.forEach((v, j) => (row[j] += lr * v));
  }
}
