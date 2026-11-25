// authored-by: Claude Opus 5.5
// Policy gradients, baselines, actor-critic and PPO's clip — the maths behind every widget in this lesson.
// Everything is tabular and tiny on purpose: each number the lesson quotes is pinned in pg.test.ts.
import { createRng, type Rng } from '../../lib/rng';

// ------------------------------------------------------------------ shared helpers

/** Scramble a small integer seed so neighbouring seeds don't give correlated first draws. */
export function mixSeed(seed: number): number {
  let x = (seed + 0x9e3779b9) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

export const seededRng = (seed: number): Rng => createRng(mixSeed(seed));

export function softmax(z: readonly number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** ∇θ log softmax(θ)[a] = e_a − π: "more of a, less of everything in proportion to how likely it was". */
export function gradLogSoftmax(theta: readonly number[], a: number): number[] {
  const p = softmax(theta);
  return p.map((pi, i) => (i === a ? 1 : 0) - pi);
}

export function logSoftmaxAt(theta: readonly number[], a: number): number {
  const m = Math.max(...theta);
  const lse = m + Math.log(theta.reduce((s, v) => s + Math.exp(v - m), 0));
  return theta[a] - lse;
}

export function sampleIndex(p: readonly number[], rng: Rng): number {
  let u = rng.next();
  for (let i = 0; i < p.length; i++) {
    u -= p[i];
    if (u < 0) return i;
  }
  return p.length - 1;
}

const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const std = (xs: readonly number[]) => {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, v) => s + (v - m) * (v - m), 0) / Math.max(1, xs.length - 1));
};

// ------------------------------------------------------------------ 1. The aliased corridor (Sutton & Barto, Example 13.1)
//
// Three grey squares that all look identical to the agent, then the goal. −1 per step.
//   square 1: Right → square 2, Left → bumps the wall, stays in 1
//   square 2: the controls are SWAPPED: Right → back to 1, Left → on to 3
//   square 3: Right → goal, Left → back to 2
// Because the squares look the same, a policy is a single number p = P(Right).

export const CORRIDOR_P_STAR = 2 - Math.SQRT2; // ≈ 0.5858

/** Exact expected return from each square under P(Right) = p (solves the 3×3 Bellman system in closed form). */
export function corridorValues(p: number): { v1: number; v2: number; v3: number } {
  if (p <= 0 || p >= 1) return { v1: -Infinity, v2: -Infinity, v3: -Infinity };
  const v2 = (p - 3) / (p * (1 - p));
  const v1 = v2 - 1 / p;
  const v3 = -1 + (1 - p) * v2;
  return { v1, v2, v3 };
}

/** Expected number of steps from the start (= −V₁). */
export const corridorSteps = (p: number) => -corridorValues(p).v1;

/** One episode: list of squares visited (1..3, then 4 = goal), capped. */
export function simulateCorridor(p: number, rng: Rng, cap = 200): { squares: number[]; actions: number[]; reached: boolean } {
  let s = 1;
  const squares = [1];
  const actions: number[] = [];
  for (let t = 0; t < cap; t++) {
    const right = rng.next() < p;
    actions.push(right ? 1 : 0);
    s = corridorStep(s, right);
    squares.push(s);
    if (s === 4) return { squares, actions, reached: true };
  }
  return { squares, actions, reached: false };
}

export function corridorStep(s: number, right: boolean): number {
  if (s === 1) return right ? 2 : 1;
  if (s === 2) return right ? 1 : 3;
  return right ? 4 : 2; // s === 3
}

/**
 * REINFORCE on the corridor with a running-average baseline. One logit θ for "Right" vs a fixed 0 for "Left",
 * so p = sigmoid(θ). Each episode's update is averaged over its steps. Returns p after every episode.
 */
export function corridorReinforce(lr: number, episodes: number, seed: number, p0 = 0.15, cap = 50): number[] {
  const rng = seededRng(seed);
  let theta = Math.log(p0 / (1 - p0));
  let baseline = 0;
  let n = 0;
  const trace = [1 / (1 + Math.exp(-theta))];
  for (let ep = 0; ep < episodes; ep++) {
    const p = 1 / (1 + Math.exp(-theta));
    const run = simulateCorridor(p, rng, cap);
    const T = run.actions.length;
    let g = 0;
    for (let t = 0; t < T; t++) {
      const G = -(T - t); // undiscounted return from step t
      const dlog = run.actions[t] === 1 ? 1 - p : -p; // d/dθ log P(action)
      g += (G - baseline) * dlog;
    }
    theta += (lr * g) / T;
    for (let t = 0; t < T; t++) {
      n++;
      baseline += (-(T - t) - baseline) / Math.min(n, 200);
    }
    trace.push(1 / (1 + Math.exp(-theta)));
  }
  return trace;
}

// ------------------------------------------------------------------ 2. The push explorer (one decision, three actions)

export const PUSH_ACTIONS = ['Dull', 'Decent', 'Great'] as const;
export const PUSH_MEANS = [1, 2, 3] as const;
export const PUSH_SIGMA = 1;

export type EstimatorStats = {
  probs: number[];
  /** Expected reward under the current policy (before any offset). */
  expectedReward: number;
  /** E[g]: the true gradient of expected reward w.r.t. the logits. */
  expected: number[];
  /** Total variance of the one-sample estimate g = (R − b)(e_a − π): E‖g‖² − ‖E g‖². */
  variance: number;
  /** For each action: the push applied to the logits when that action is sampled and pays its average. */
  pushes: number[][];
};

/** Exact statistics of the score-function gradient estimate for a one-step bandit. */
export function estimatorStats(
  theta: readonly number[],
  means: readonly number[],
  sigma: number,
  offset: number,
  baseline: number,
): EstimatorStats {
  const p = softmax(theta);
  const k = p.length;
  const expected = new Array<number>(k).fill(0);
  let e2 = 0;
  const pushes: number[][] = [];
  for (let a = 0; a < k; a++) {
    const g = p.map((pi, i) => (i === a ? 1 : 0) - pi);
    const w = means[a] + offset - baseline;
    pushes.push(g.map((v) => w * v));
    for (let i = 0; i < k; i++) expected[i] += p[a] * w * g[i];
    e2 += p[a] * (w * w + sigma * sigma) * g.reduce((s, v) => s + v * v, 0);
  }
  const variance = e2 - expected.reduce((s, v) => s + v * v, 0);
  const expectedReward = p.reduce((s, pi, i) => s + pi * means[i], 0);
  return { probs: p, expectedReward, expected, variance, pushes };
}

/** Same statistics with the best simple baseline: the policy's own expected reward (including the offset). */
export function estimatorStatsWithBaseline(theta: readonly number[], means: readonly number[], sigma: number, offset: number) {
  const p = softmax(theta);
  const b = p.reduce((s, pi, i) => s + pi * means[i], 0) + offset;
  return { baseline: b, ...estimatorStats(theta, means, sigma, offset, b) };
}

// ------------------------------------------------------------------ 3. The long noisy hallway: REINFORCE vs baseline vs actor-critic

export const HALL = { cells: 8, stepReward: -0.5, noise: 2, goal: 10, gamma: 0.95, cap: 40 } as const;

/** Return of walking straight to the goal (the best any policy can expect). */
export function hallwayOptimalReturn(): number {
  let ret = 0;
  let d = 1;
  for (let t = 0; t < HALL.cells; t++) {
    ret += d * (HALL.stepReward + (t === HALL.cells - 1 ? HALL.goal : 0));
    d *= HALL.gamma;
  }
  return ret;
}

export type Estimator = 'none' | 'average' | 'state' | 'td';
export const ESTIMATOR_LABELS: Record<Estimator, string> = {
  none: 'No baseline',
  average: 'One running average',
  state: 'Per-state V(s)',
  td: 'Actor-critic (TD)',
};

/**
 * Learning rate for every baseline and critic: a plain average of everything seen for the first 1/β samples,
 * then an exponential moving average with rate β (so it starts from real data rather than from 0).
 */
export const criticRate = (beta: number, n: number) => Math.max(beta, 1 / n);

export type HallRun = { returns: number[]; pRight: number[]; advantages: number[] };

/**
 * Train a tabular softmax policy (2 logits per cell: Left, Right) in the hallway.
 *  - none:    θ += α · G_t · ∇log π          (REINFORCE)
 *  - average: θ += α · (G_t − b̄) · ∇log π   (b̄ = running average of all returns, rate β)
 *  - state:   θ += α · (G_t − V(s_t)) · ∇log π (V learned from Monte-Carlo returns, rate β)
 *  - td:      θ += α · δ_t · ∇log π, δ_t = r + γV(s′) − V(s), updated online every step (actor-critic)
 * Monte-Carlo methods apply their update after the episode using the policy that acted.
 */
export function runHallway(est: Estimator, alpha: number, beta: number, episodes: number, seed: number): HallRun {
  const { cells, stepReward, noise, goal, gamma, cap } = HALL;
  const rng = seededRng(seed);
  const theta = Array.from({ length: cells }, () => [0, 0]);
  const V = new Array<number>(cells).fill(0);
  const visits = new Array<number>(cells).fill(0);
  let avg = 0;
  let avgN = 0;
  const returns: number[] = [];
  const advantages: number[] = [];
  for (let ep = 0; ep < episodes; ep++) {
    let s = 0;
    let ret = 0;
    let disc = 1;
    const S: number[] = [];
    const A: number[] = [];
    const R: number[] = [];
    const P: number[][] = [];
    for (let t = 0; t < cap; t++) {
      const p = softmax(theta[s]);
      const a = sampleIndex(p, rng);
      const ns = a === 1 ? s + 1 : Math.max(0, s - 1);
      const done = ns === cells;
      const r = stepReward + rng.normal(0, noise) + (done ? goal : 0);
      ret += disc * r;
      disc *= gamma;
      if (est === 'td') {
        const delta = r + (done ? 0 : gamma * V[ns]) - V[s];
        V[s] += criticRate(beta, ++visits[s]) * delta;
        advantages.push(delta);
        for (let i = 0; i < 2; i++) theta[s][i] += alpha * delta * ((i === a ? 1 : 0) - p[i]);
      } else {
        S.push(s);
        A.push(a);
        R.push(r);
        P.push(p);
      }
      if (done) break;
      s = ns;
    }
    if (est !== 'td') {
      let G = 0;
      const Gs = new Array<number>(S.length);
      for (let t = S.length - 1; t >= 0; t--) {
        G = R[t] + gamma * G;
        Gs[t] = G;
      }
      for (let t = 0; t < S.length; t++) {
        const s0 = S[t];
        const b = est === 'none' ? 0 : est === 'average' ? avg : V[s0];
        const adv = Gs[t] - b;
        advantages.push(adv);
        for (let i = 0; i < 2; i++) theta[s0][i] += alpha * adv * ((i === A[t] ? 1 : 0) - P[t][i]);
      }
      for (let t = 0; t < S.length; t++) {
        if (est === 'average') avg += criticRate(beta, ++avgN) * (Gs[t] - avg);
        if (est === 'state') V[S[t]] += criticRate(beta, ++visits[S[t]]) * (Gs[t] - V[S[t]]);
      }
    }
    returns.push(ret);
  }
  return { returns, pRight: theta.map((th) => softmax(th)[1]), advantages };
}

export const RACE_EPISODES = 150;
export const RACE_SEEDS = 20;
export const RACE_ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] as const;
export const RACE_BETAS = [0.05, 0.1, 0.2, 0.5] as const;
export const RACE_METHODS: Estimator[] = ['none', 'state', 'td'];

export type RaceEntry = { est: Estimator; alpha: number; beta: number; score: number; scores: number[]; late: number; curve: number[]; advSpread: number };

/** Mean return over every training episode (area under the learning curve), per seed. */
export function hallwayScores(est: Estimator, alpha: number, beta: number, episodes = RACE_EPISODES, seeds = RACE_SEEDS) {
  const scores: number[] = [];
  const lates: number[] = [];
  const advSpread: number[] = [];
  const curve = new Array<number>(episodes).fill(0);
  for (let sd = 1; sd <= seeds; sd++) {
    const run = runHallway(est, alpha, beta, episodes, sd);
    scores.push(mean(run.returns));
    lates.push(mean(run.returns.slice(-50)));
    advSpread.push(std(run.advantages));
    run.returns.forEach((r, i) => (curve[i] += r / seeds));
  }
  return { scores, score: mean(scores), late: mean(lates), curve, advSpread: mean(advSpread) };
}

/** Grid search: each method at its own best (α, β) by area under the learning curve. Used by tests; results recorded below. */
export function searchHallwayRace(): RaceEntry[] {
  return RACE_METHODS.map((est) => {
    let best: RaceEntry | null = null;
    for (const alpha of RACE_ALPHAS)
      for (const beta of est === 'none' ? [0] : RACE_BETAS) {
        const r = hallwayScores(est, alpha, beta);
        if (!best || r.score > best.score) best = { est, alpha, beta, ...r };
      }
    return best!;
  });
}

/** The recorded result of searchHallwayRace() (re-verified in pg.test.ts). */
export const RACE_BEST: { est: Estimator; alpha: number; beta: number }[] = [
  { est: 'none', alpha: 0.05, beta: 0 },
  { est: 'state', alpha: 0.05, beta: 0.1 },
  { est: 'td', alpha: 0.1, beta: 0.2 },
];

export function raceSummary() {
  return RACE_BEST.map((b) => ({ ...b, ...hallwayScores(b.est, b.alpha, b.beta) }));
}

export { mean as meanOf, std as stdOf };

// ------------------------------------------------------------------ 4. PPO's clipped surrogate

export const PPO_MEANS = [1, 2, 3] as const;
export const PPO_NOISE = 2;
export const PPO_BATCH = 8;

/** The clipped objective for one sample: min(r·A, clip(r, 1−ε, 1+ε)·A). */
export function clippedObjective(ratio: number, adv: number, eps: number): number {
  const clipped = Math.min(Math.max(ratio, 1 - eps), 1 + eps);
  return Math.min(ratio * adv, clipped * adv);
}

/** Is the gradient of the clipped objective switched off for this sample? */
export function clipInactive(ratio: number, adv: number, eps: number): boolean {
  return (adv > 0 && ratio > 1 + eps) || (adv < 0 && ratio < 1 - eps);
}

export type Batch = { actions: number[]; rewards: number[]; advantages: number[] };

export function drawBatch(theta: readonly number[], rng: Rng, size = PPO_BATCH): Batch {
  const p = softmax(theta);
  const actions: number[] = [];
  const rewards: number[] = [];
  for (let n = 0; n < size; n++) {
    const a = sampleIndex(p, rng);
    actions.push(a);
    rewards.push(rng.normal(PPO_MEANS[a], PPO_NOISE));
  }
  const m = mean(rewards);
  return { actions, rewards, advantages: rewards.map((r) => r - m) };
}

/**
 * K full-batch gradient-ascent epochs on the surrogate mean(r·A) (clip = null) or PPO's clipped version.
 * Returns the logits after every epoch.
 */
export function reuseBatch(theta0: readonly number[], batch: Batch, epochs: number, lr: number, clip: number | null): number[][] {
  const pOld = softmax(theta0);
  const th = theta0.slice();
  const trace = [th.slice()];
  const k = th.length;
  for (let e = 0; e < epochs; e++) {
    const p = softmax(th);
    const g = new Array<number>(k).fill(0);
    for (let n = 0; n < batch.actions.length; n++) {
      const a = batch.actions[n];
      const A = batch.advantages[n];
      const r = p[a] / pOld[a];
      if (clip !== null && clipInactive(r, A, clip)) continue;
      for (let i = 0; i < k; i++) g[i] += (A * r * ((i === a ? 1 : 0) - p[i])) / batch.actions.length;
    }
    for (let i = 0; i < k; i++) th[i] += lr * g[i];
    trace.push(th.slice());
  }
  return trace;
}

export const expectedPPOReward = (theta: readonly number[]) => softmax(theta).reduce((s, p, i) => s + p * PPO_MEANS[i], 0);

/** A short training run: `iters` batches, each reused for `epochs` epochs. Returns the final expected reward. */
export function ppoTrain(epochs: number, lr: number, clip: number | null, iters: number, seed: number): number {
  const rng = seededRng(seed);
  let th = [0, 0, 0];
  for (let it = 0; it < iters; it++) {
    const batch = drawBatch(th, rng);
    const tr = reuseBatch(th, batch, epochs, lr, clip);
    th = tr[tr.length - 1];
  }
  return expectedPPOReward(th);
}

export const PPO_LRS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5] as const;
export const PPO_SEEDS = 100;
export const PPO_ITERS = 20;
export const PPO_FAIL = 2.5;

/** For each step size: mean final expected reward and how many of 100 seeds ended below 2.5 (stuck off the best action). */
export function ppoRobustness(epochs: number, clip: number | null) {
  return PPO_LRS.map((lr) => {
    const finals: number[] = [];
    for (let s = 1; s <= PPO_SEEDS; s++) finals.push(ppoTrain(epochs, lr, clip, PPO_ITERS, s));
    return { lr, mean: mean(finals), failures: finals.filter((f) => f < PPO_FAIL).length };
  });
}

// ------------------------------------------------------------------ 5. Lab: stabilise the policy gradient

export type LabBriefId = 'paid' | 'payscales' | 'hallway';
export type LabConfig = { est: Estimator; alpha: number };

export const LAB_ESTIMATORS: Estimator[] = ['none', 'average', 'state', 'td'];
export const LAB_ALPHAS = [0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5] as const;
export const LAB_BETA = 0.1;
export const LAB_SEEDS = 20;

export type LabBrief = {
  id: LabBriefId;
  title: string;
  story: string;
  metric: string;
  threshold: number;
  budget: string;
};

export const LAB_BRIEFS: LabBrief[] = [
  {
    id: 'paid',
    title: 'Brief 1 · Everybody gets paid',
    story:
      'Three buttons on a vending machine that always pays out about 50 coins: 49, 50 or 51 on average, give or take 1. Learn to press the 51-coin button.',
    metric: 'average probability of the best button after 300 presses',
    threshold: 0.8,
    budget: '300 presses',
  },
  {
    id: 'payscales',
    title: 'Brief 2 · Four rooms, four payscales',
    story:
      'The same three buttons appear in four rooms, but each room adds its own bonus (0, +20, −20 or +40 coins). In every room the third button is still 1 coin better than the second, which is 1 better than the first.',
    metric: 'probability of the best button, averaged over the four rooms, after 600 presses',
    threshold: 0.75,
    budget: '600 presses',
  },
  {
    id: 'hallway',
    title: 'Brief 3 · The long noisy hallway',
    story:
      'Eight squares to the exit (+10). Every step costs 0.5 on average, but each step’s reward is jittered by noise of ±2. Discount γ = 0.95, at most 40 steps per episode.',
    metric: 'average return over the last 50 of 150 episodes',
    threshold: 3.0,
    budget: '150 episodes',
  },
];

const PAID = { offsets: [50], deltas: [-1, 0, 1], sigma: 1, pulls: 300 };
const PAYSCALES = { offsets: [0, 20, -20, 40], deltas: [-1, 0, 1], sigma: 1, pulls: 600 };

/** Contextual bandit: each press, a room is drawn at random; the agent presses a button and learns from the reward. */
export function runContextBandit(
  env: { offsets: readonly number[]; deltas: readonly number[]; sigma: number; pulls: number },
  est: Estimator,
  alpha: number,
  beta: number,
  seed: number,
): { pBest: number[]; curve: number[] } {
  const rng = seededRng(seed);
  const C = env.offsets.length;
  const K = env.deltas.length;
  const best = env.deltas.indexOf(Math.max(...env.deltas));
  const th = Array.from({ length: C }, () => new Array<number>(K).fill(0));
  const V = new Array<number>(C).fill(0);
  const visits = new Array<number>(C).fill(0);
  let avg = 0;
  const curve: number[] = [];
  for (let t = 0; t < env.pulls; t++) {
    const c = rng.int(C);
    const p = softmax(th[c]);
    const a = sampleIndex(p, rng);
    const r = env.offsets[c] + env.deltas[a] + rng.normal(0, env.sigma);
    // In a one-step problem the TD error r − V(s) is exactly the per-state advantage.
    const b = est === 'none' ? 0 : est === 'average' ? avg : V[c];
    const adv = r - b;
    for (let i = 0; i < K; i++) th[c][i] += alpha * adv * ((i === a ? 1 : 0) - p[i]);
    avg += criticRate(beta, t + 1) * (r - avg);
    V[c] += criticRate(beta, ++visits[c]) * (r - V[c]);
    curve.push(mean(th.map((row) => softmax(row)[best])));
  }
  return { pBest: th.map((row) => softmax(row)[best]), curve };
}

export type LabResult = { score: number; perSeed: number[]; curve: number[]; pass: boolean };

const labCache = new Map<string, LabResult>();

export function evaluateLab(brief: LabBriefId, cfg: LabConfig): LabResult {
  const key = `${brief}|${cfg.est}|${cfg.alpha}`;
  const hit = labCache.get(key);
  if (hit) return hit;
  const b = LAB_BRIEFS.find((x) => x.id === brief)!;
  const perSeed: number[] = [];
  let curve: number[] = [];
  for (let sd = 1; sd <= LAB_SEEDS; sd++) {
    let s: number;
    let c: number[];
    if (brief === 'hallway') {
      const run = runHallway(cfg.est, cfg.alpha, LAB_BETA, 150, sd);
      s = mean(run.returns.slice(-50));
      c = run.returns;
    } else {
      const run = runContextBandit(brief === 'paid' ? PAID : PAYSCALES, cfg.est, cfg.alpha, LAB_BETA, sd);
      s = mean(run.pBest);
      c = run.curve;
    }
    perSeed.push(s);
    if (curve.length === 0) curve = new Array<number>(c.length).fill(0);
    c.forEach((v, i) => (curve[i] += v / LAB_SEEDS));
  }
  const score = mean(perSeed);
  const res = { score, perSeed, curve, pass: score >= b.threshold };
  labCache.set(key, res);
  return res;
}

export function solveLabExhaustive(brief: LabBriefId) {
  const all: { cfg: LabConfig; score: number; pass: boolean }[] = [];
  for (const est of LAB_ESTIMATORS)
    for (const alpha of LAB_ALPHAS) {
      const r = evaluateLab(brief, { est, alpha });
      all.push({ cfg: { est, alpha }, score: r.score, pass: r.pass });
    }
  const passing = all.filter((x) => x.pass);
  const best = all.reduce((a, b) => (b.score > a.score ? b : a));
  return { total: all.length, passing, best, all };
}

// ------------------------------------------------------------------ sign errors of single samples

/** Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
export function normalCdf(z: number): number {
  const t = 1 / (1 + 0.3275911 * (Math.abs(z) / Math.SQRT2));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/**
 * Probability that ONE sampled estimate pushes logit `idx` in the opposite direction to the true gradient.
 * The push on logit idx is (R − b)(1[a = idx] − π_idx), with R ~ Normal(mean_a + offset, σ).
 */
export function wrongWayProbability(
  theta: readonly number[],
  means: readonly number[],
  sigma: number,
  offset: number,
  baseline: number,
  idx: number,
): number {
  const st = estimatorStats(theta, means, sigma, offset, baseline);
  const trueSign = Math.sign(st.expected[idx]);
  const p = st.probs;
  let wrong = 0;
  for (let a = 0; a < p.length; a++) {
    const factor = (a === idx ? 1 : 0) - p[idx]; // sign of the push is sign(R − b) × sign(factor)
    const pPos = 1 - normalCdf((baseline - (means[a] + offset)) / sigma); // P(R − b > 0)
    const pushPositive = factor > 0 ? pPos : 1 - pPos;
    wrong += p[a] * (trueSign > 0 ? 1 - pushPositive : pushPositive);
  }
  return wrong;
}
