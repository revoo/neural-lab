// authored-by: Gemini Antigravity
/**
 * Exact statistical machinery for AI system evaluation:
 *  - Exact Binomial PMF/CDF (log-factorial, self-contained)
 *  - Paired (exact McNemar) vs Unpaired (exact two-binomial) A/B comparisons
 *  - Exact Winner's Curse / eval overfitting distribution for max of K independent runs
 *  - Stated LLM-as-a-judge position bias model (single-order vs position-swapped)
 *  - Exact binomial power, false-positive rate and cost for Lab: Size Your Eval Suite
 */
import { createRng } from '../../lib/rng';

// ------------------------------------------------------------------ exact binomial primitives

const LOG_FACT_CACHE: number[] = [0, 0];
function logFact(n: number): number {
  while (LOG_FACT_CACHE.length <= n) {
    const k = LOG_FACT_CACHE.length;
    LOG_FACT_CACHE.push(LOG_FACT_CACHE[k - 1] + Math.log(k));
  }
  return LOG_FACT_CACHE[n];
}

/** Exact PMF array for Binomial(n, p): out[k] = P(X = k) for k = 0..n. */
export function binomialPmf(n: number, p: number): number[] {
  const out = new Array<number>(n + 1).fill(0);
  if (p <= 0) {
    out[0] = 1;
    return out;
  }
  if (p >= 1) {
    out[n] = 1;
    return out;
  }
  const lp = Math.log(p);
  const lq = Math.log(1 - p);
  const ln = logFact(n);
  for (let k = 0; k <= n; k++) {
    out[k] = Math.exp(ln - logFact(k) - logFact(n - k) + k * lp + (n - k) * lq);
  }
  return out;
}

/** Exact upper-tail probability P(Binomial(n, p) >= k0). */
export function binomialUpperTail(n: number, p: number, k0: number): number {
  if (k0 <= 0) return 1;
  if (k0 > n) return 0;
  const pmf = binomialPmf(n, p);
  let s = 0;
  for (let k = k0; k <= n; k++) s += pmf[k];
  return Math.min(1, s);
}

/** Standard error of a single proportion: sqrt(p(1-p)/n). */
export const propSE = (p: number, n: number) => Math.sqrt((p * (1 - p)) / n);

// ------------------------------------------------------------------ 1. Paired vs Unpaired comparison

/** Exact two-sided McNemar p-value from discordant counts (n10 = only A right, n01 = only B right). */
export function mcnemarExactP(n10: number, n01: number): number {
  const m = n10 + n01;
  if (m === 0 || n10 === n01) return 1;
  const hi = Math.max(n10, n01);
  return Math.min(1, 2 * binomialUpperTail(m, 0.5, hi));
}

/**
 * Exact two-sided p-value for two independent Binomial(n, pHat) draws showing a gap >= |kA - kB|,
 * evaluated under the pooled null rate pHat = (kA + kB) / (2n).
 */
export function unpairedExactP(n: number, kA: number, kB: number): number {
  const diff = Math.abs(kA - kB);
  if (diff === 0) return 1;
  const pHat = (kA + kB) / (2 * n);
  const pmf = binomialPmf(n, pHat);
  const cdf = new Array<number>(n + 2).fill(0);
  for (let k = 0; k <= n; k++) cdf[k + 1] = cdf[k] + pmf[k];
  let oneSided = 0;
  for (let ka = diff; ka <= n; ka++) {
    oneSided += pmf[ka] * cdf[ka - diff + 1];
  }
  return Math.min(1, 2 * oneSided);
}

export function pairedComparison(n: number, kA: number, kB: number, n11: number) {
  const min11 = Math.max(0, kA + kB - n);
  const max11 = Math.min(kA, kB);
  const c11 = Math.max(min11, Math.min(max11, n11));
  const n10 = kA - c11;
  const n01 = kB - c11;
  const n00 = n - c11 - n10 - n01;
  const m = n10 + n01;
  return {
    n,
    kA,
    kB,
    n11: c11,
    n10,
    n01,
    n00,
    min11,
    max11,
    disagreements: m,
    accA: kA / n,
    accB: kB / n,
    gap: (kA - kB) / n,
    mcnemarP: mcnemarExactP(n10, n01),
    unpairedP: unpairedExactP(n, kA, kB),
    /** Exact fair-coin distribution over n10 given m disagreements. */
    coinPmf: m > 0 ? binomialPmf(m, 0.5) : [1],
  };
}

// ------------------------------------------------------------------ 2. Winner's Curse (eval overfitting)

/**
 * Exact distribution of the maximum of K independent Binomial(n, p) prompt scores on an n-question eval.
 * F_max(k) = F(k)^K, so P(max = k) = F(k)^K - F(k-1)^K.
 */
export function winnersCurse(n: number, p: number, K: number) {
  const single = binomialPmf(n, p);
  const maxPmf = new Array<number>(n + 1).fill(0);
  let cdf = 0;
  let prevPow = 0;
  let expectedMax = 0;
  let probPlus5 = 0;
  const need5 = Math.ceil((p + 0.05) * n - 1e-9);

  for (let k = 0; k <= n; k++) {
    cdf = Math.min(1, cdf + single[k]);
    const pow = cdf ** K;
    const pk = Math.max(0, pow - prevPow);
    maxPmf[k] = pk;
    prevPow = pow;
    expectedMax += pk * (k / n);
    if (k >= need5) probPlus5 += pk;
  }

  return {
    single,
    maxPmf,
    trueAcc: p,
    expectedMax,
    expectedMirage: expectedMax - p,
    probPlus5,
  };
}

// ------------------------------------------------------------------ 3. LLM-as-a-Judge & Position Bias

export type JudgeProtocol = 'a-first' | 'b-first' | 'swapped';

/**
 * Stated probabilistic model of a pairwise LLM judge:
 *  - Ground truth split: A truly better (trueWinA), B truly better (trueWinB), true tie (1 - trueWinA - trueWinB).
 *  - Judge base accuracy on clear wins is `q` (in [0.5, 1]).
 *  - Position bias `b` (in [0, 0.4]) favours Slot 1:
 *      * Slot 1 true winner picked with prob q1 = min(1, q + b/2)
 *      * Slot 2 true winner picked with prob q2 = max(0, q - b/2)
 *      * True tie picked as Slot 1 with prob 0.5 + b, Slot 2 with prob 0.5 - b.
 *  - Under `swapped`, both (A, B) and (B, A) are judged independently; a win counts only if both
 *    orderings agree, otherwise the pair is recorded as a flip / tie.
 */
export function judgeModel(trueWinA: number, trueWinB: number, q: number, b: number) {
  const tie = Math.max(0, 1 - trueWinA - trueWinB);
  const q1 = Math.min(1, q + b / 2);
  const q2 = Math.max(0, q - b / 2);

  // Single order: A in Slot 1, B in Slot 2
  const aFirstWinA = trueWinA * q1 + trueWinB * (1 - q2) + tie * (0.5 + b);
  const aFirstWinB = 1 - aFirstWinA;

  // Single order: B in Slot 1, A in Slot 2
  const bFirstWinB = trueWinB * q1 + trueWinA * (1 - q2) + tie * (0.5 + b);
  const bFirstWinA = 1 - bFirstWinB;

  // Swapped (both orders independently): consistent A win requires A wins in Slot 1 AND A wins in Slot 2
  const swapWinA = trueWinA * q1 * q2 + trueWinB * (1 - q2) * (1 - q1) + tie * (0.5 + b) * (0.5 - b);
  const swapWinB = trueWinB * q1 * q2 + trueWinA * (1 - q2) * (1 - q1) + tie * (0.5 + b) * (0.5 - b);
  const swapFlip = Math.max(0, 1 - swapWinA - swapWinB);

  return {
    trueWinA,
    trueWinB,
    tie,
    q1,
    q2,
    aFirst: { winA: aFirstWinA, winB: aFirstWinB, flip: 0, netA: aFirstWinA - aFirstWinB },
    bFirst: { winA: bFirstWinA, winB: bFirstWinB, flip: 0, netA: bFirstWinA - bFirstWinB },
    swapped: { winA: swapWinA, winB: swapWinB, flip: swapFlip, netA: swapWinA - swapWinB },
  };
}

// ------------------------------------------------------------------ 4. Lab: Size Your Eval Suite

export type EvalMode = 'unpaired' | 'paired';
export type GraderId = 'human' | 'judge1' | 'judge2';

export type GraderSpec = {
  id: GraderId;
  label: string;
  shortLabel: string;
  accuracy: number; // symmetric binary grading accuracy q
  costPerItem: number; // credits per graded item
};

export const GRADERS: GraderSpec[] = [
  { id: 'human', label: 'Exact human / unit test (100% acc, 5.0 cr)', shortLabel: 'Exact (5cr)', accuracy: 1.0, costPerItem: 5.0 },
  { id: 'judge2', label: 'Dual-pass LLM judge (93% acc, 1.0 cr)', shortLabel: 'Dual LLM (1cr)', accuracy: 0.93, costPerItem: 1.0 },
  { id: 'judge1', label: 'Fast LLM judge (85% acc, 0.4 cr)', shortLabel: 'Fast LLM (0.4cr)', accuracy: 0.85, costPerItem: 0.4 },
];

export const GRADER_BY_ID = Object.fromEntries(GRADERS.map((g) => [g.id, g])) as Record<GraderId, GraderSpec>;

export const LAB_NS = [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000] as const;
export const LAB_ALPHAS = [0.1, 0.05, 0.02, 0.01] as const;
export const LAB_MODES: EvalMode[] = ['unpaired', 'paired'];

export type EvalDesign = {
  mode: EvalMode;
  grader: GraderId;
  n: number;
  alpha: number;
};

export type EvalBrief = {
  id: 'release' | 'tight-set' | 'fine-gain';
  title: string;
  story: string;
  p0: number; // baseline true accuracy
  delta: number; // candidate true gain (+delta)
  baseDiscord: number; // under paired H0, prob only A right = prob only B right = baseDiscord
  minPower: number; // minimum detection power for +delta
  maxFpr: number; // maximum false-positive rate when delta = 0
  maxN?: number; // optional hard cap on available test questions
};

export const LAB_BRIEFS: EvalBrief[] = [
  {
    id: 'release',
    title: 'Brief 1 · Routine Release Gate',
    story: 'You want to detect a real +5.0% accuracy gain over an 80% baseline at least 85% of the time (power ≥ 85%), while shipping a no-op change at most 5% of the time (false positives ≤ 5%). Any number of test items up to 2,000 is available.',
    p0: 0.8,
    delta: 0.05,
    baseDiscord: 0.04,
    minPower: 0.85,
    maxFpr: 0.05,
  },
  {
    id: 'tight-set',
    title: 'Brief 2 · Capped Benchmark (n ≤ 1,000)',
    story: 'You have at most 1,000 curated support tickets (n ≤ 1,000). Still detect a +5.0% gain over an 80% baseline with power ≥ 85% and false positives ≤ 5% as cheaply as possible.',
    p0: 0.8,
    delta: 0.05,
    baseDiscord: 0.04,
    minPower: 0.85,
    maxFpr: 0.05,
    maxN: 1000,
  },
  {
    id: 'fine-gain',
    title: 'Brief 3 · High-Stakes +3% Lift',
    story: 'A subtle prompt overhaul yields a real +3.0% gain over an 85% baseline. Detect it with power ≥ 85% and strict false-positive rate ≤ 2% at the lowest total grading cost.',
    p0: 0.85,
    delta: 0.03,
    baseDiscord: 0.03,
    minPower: 0.85,
    maxFpr: 0.02,
  },
];

/**
 * Exact probability P(K1 - K0 >= thresh) for independent K1 ~ Bin(n, p1), K0 ~ Bin(n, p0).
 */
function independentDiffTail(n: number, p1: number, p0: number, thresh: number): number {
  const pmf1 = binomialPmf(n, p1);
  const pmf0 = binomialPmf(n, p0);
  const cdf0 = new Array<number>(n + 2).fill(0);
  for (let k = 0; k <= n; k++) cdf0[k + 1] = cdf0[k] + pmf0[k];
  let s = 0;
  for (let k1 = 0; k1 <= n; k1++) {
    const maxK0 = k1 - thresh;
    if (maxK0 < 0) continue;
    s += pmf1[k1] * cdf0[Math.min(n, maxK0) + 1];
  }
  return Math.min(1, s);
}

/** Cache for exact one-sided McNemar critical values c_alpha(m) = min c s.t. P(Bin(m, 0.5) >= c) <= alpha. */
const MCNEMAR_CRIT_CACHE = new Map<string, Int32Array>();

function mcnemarCritTable(maxM: number, alpha: number): Int32Array {
  const key = `${maxM}:${alpha}`;
  const hit = MCNEMAR_CRIT_CACHE.get(key);
  if (hit) return hit;
  const table = new Int32Array(maxM + 1);
  table[0] = 1; // impossible to reach 1 when m = 0
  for (let m = 1; m <= maxM; m++) {
    const pmf = binomialPmf(m, 0.5);
    let tail = 0;
    let crit = m + 1;
    for (let k = m; k >= 0; k--) {
      if (tail + pmf[k] <= alpha + 1e-12) {
        tail += pmf[k];
        crit = k;
      } else {
        break;
      }
    }
    table[m] = crit;
  }
  MCNEMAR_CRIT_CACHE.set(key, table);
  return table;
}

/**
 * Evaluate exact False Positive Rate (at delta = 0), exact Detection Power (at +delta),
 * attenuated observed gap, and total grading cost.
 *
 * Cost model:
 *  - Unpaired: baseline score is already known from previous split; you grade n new items for the
 *    candidate (1.0 * n * graderCost).
 *  - Paired: you run and grade BOTH baseline and candidate fresh on the same n items (1.5 * n * graderCost,
 *    reflecting running & grading both outputs on the shared set).
 */
export function evaluateEvalDesign(d: EvalDesign, b: EvalBrief) {
  const g = GRADER_BY_ID[d.grader];
  const q = g.accuracy;
  const p0 = b.p0;
  const p1 = Math.min(0.999, b.p0 + b.delta);

  // Observed accuracies through symmetric noisy grader q
  const obs0 = p0 * q + (1 - p0) * (1 - q);
  const obs1 = p1 * q + (1 - p1) * (1 - q);
  const observedGap = obs1 - obs0; // = delta * (2q - 1)

  if (d.mode === 'unpaired') {
    // Find smallest integer threshold t such that P_H0(K1 - K0 >= t) <= alpha
    // Binary search t in [0..n]
    let lo = 0;
    let hi = d.n + 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (independentDiffTail(d.n, obs0, obs0, mid) <= d.alpha + 1e-12) hi = mid;
      else lo = mid + 1;
    }
    const thresh = lo;
    const fpr = thresh > d.n ? 0 : independentDiffTail(d.n, obs0, obs0, thresh);
    const power = thresh > d.n ? 0 : independentDiffTail(d.n, obs1, obs0, thresh);
    const cost = Math.round(d.n * g.costPerItem * 10) / 10;
    return { fpr, power, observedGap, cost, thresh, effectiveDiscord: null as number | null };
  }

  // Paired mode: exact 2x2 joint probabilities under H0 and H1, passed through independent grader q
  // H0: p10 = p01 = baseDiscord, p11 = p0 - baseDiscord, p00 = 1 - p0 - baseDiscord
  // H1: candidate improves by delta -> p10 = baseDiscord + delta, p01 = baseDiscord (so p1 = p0 + delta)
  const pairedProbs = (trueP10: number, trueP01: number) => {
    const agree = 1 - trueP10 - trueP01;
    const obsP10 = trueP10 * q * q + trueP01 * (1 - q) * (1 - q) + agree * q * (1 - q);
    const obsP01 = trueP01 * q * q + trueP10 * (1 - q) * (1 - q) + agree * q * (1 - q);
    const pd = obsP10 + obsP01;
    const theta = pd > 0 ? obsP10 / pd : 0.5;
    return { obsP10, obsP01, pd, theta };
  };

  const h0 = pairedProbs(b.baseDiscord, b.baseDiscord);
  const h1 = pairedProbs(b.baseDiscord + b.delta, b.baseDiscord);

  // Sum over M ~ Bin(n, pd) with exact conditional Bin(M, theta) >= crit(M)
  const crit = mcnemarCritTable(d.n, d.alpha);

  const exactMcNemarRate = (pd: number, theta: number) => {
    const pmfM = binomialPmf(d.n, pd);
    let total = 0;
    for (let m = 1; m <= d.n; m++) {
      if (pmfM[m] < 1e-15) continue;
      const c = crit[m];
      if (c > m) continue;
      total += pmfM[m] * binomialUpperTail(m, theta, c);
    }
    return Math.min(1, total);
  };

  const fpr = exactMcNemarRate(h0.pd, 0.5);
  const power = exactMcNemarRate(h1.pd, h1.theta);
  const cost = Math.round(1.5 * d.n * g.costPerItem * 10) / 10;
  return { fpr, power, observedGap, cost, thresh: null as number | null, effectiveDiscord: h1.pd };
}

/** Seeded Monte Carlo verifier for `evaluateEvalDesign` (used in unit tests). */
export function simulateEvalDesign(d: EvalDesign, b: EvalBrief, trials: number, seed = 77) {
  const rng = createRng(seed);
  const g = GRADER_BY_ID[d.grader];
  const q = g.accuracy;
  const ev = evaluateEvalDesign(d, b);

  if (d.mode === 'unpaired') {
    const obs0 = b.p0 * q + (1 - b.p0) * (1 - q);
    const obs1 = (b.p0 + b.delta) * q + (1 - b.p0 - b.delta) * (1 - q);
    const thresh = ev.thresh!;
    let fp = 0;
    let tp = 0;
    for (let t = 0; t < trials; t++) {
      let k0a = 0;
      let k0b = 0;
      let k1 = 0;
      for (let i = 0; i < d.n; i++) {
        if (rng.next() < obs0) k0a++;
        if (rng.next() < obs0) k0b++;
        if (rng.next() < obs1) k1++;
      }
      if (k0b - k0a >= thresh) fp++;
      if (k1 - k0a >= thresh) tp++;
    }
    return { fpr: fp / trials, power: tp / trials };
  }

  const crit = mcnemarCritTable(d.n, d.alpha);
  const p10_1 = b.baseDiscord + b.delta;
  const p01_1 = b.baseDiscord;
  const p11_1 = b.p0 - b.baseDiscord;
  let tp = 0;
  for (let t = 0; t < trials; t++) {
    let obs10 = 0;
    let obs01 = 0;
    for (let i = 0; i < d.n; i++) {
      const u = rng.next();
      let a = 0;
      let bb = 0;
      if (u < p11_1) {
        a = 1;
        bb = 1;
      } else if (u < p11_1 + p10_1) {
        a = 1;
        bb = 0;
      } else if (u < p11_1 + p10_1 + p01_1) {
        a = 0;
        bb = 1;
      }
      const ga = rng.next() < q ? a : 1 - a;
      const gb = rng.next() < q ? bb : 1 - bb;
      if (ga === 1 && gb === 0) obs10++;
      else if (ga === 0 && gb === 1) obs01++;
    }
    const m = obs10 + obs01;
    if (m > 0 && obs10 >= crit[m]) tp++;
  }
  return { fpr: ev.fpr, power: tp / trials };
}

export function briefPasses(b: EvalBrief, d: EvalDesign, r: ReturnType<typeof evaluateEvalDesign>): boolean {
  if (b.maxN !== undefined && d.n > b.maxN) return false;
  if (r.power < b.minPower - 1e-9) return false;
  if (r.fpr > b.maxFpr + 1e-9) return false;
  return true;
}

const EXHAUSTIVE_CACHE = new Map<string, ReturnType<typeof computeLabExhaustive>>();

function computeLabExhaustive(b: EvalBrief) {
  let total = 0;
  let passing = 0;
  let bestCost = Infinity;
  let best: { design: EvalDesign; eval: ReturnType<typeof evaluateEvalDesign> }[] = [];

  for (const mode of LAB_MODES)
    for (const grader of GRADERS.map((g) => g.id))
      for (const n of LAB_NS) {
        if (b.maxN !== undefined && n > b.maxN) continue;
        for (const alpha of LAB_ALPHAS) {
          total++;
          const design: EvalDesign = { mode, grader, n, alpha };
          const ev = evaluateEvalDesign(design, b);
          if (!briefPasses(b, design, ev)) continue;
          passing++;
          if (ev.cost < bestCost - 1e-9) {
            bestCost = ev.cost;
            best = [{ design, eval: ev }];
          } else if (Math.abs(ev.cost - bestCost) < 1e-9) {
            best.push({ design, eval: ev });
          }
        }
      }

  return { total, passing, bestCost, best };
}

export function labExhaustive(b: EvalBrief) {
  const hit = EXHAUSTIVE_CACHE.get(b.id);
  if (hit) return hit;
  const res = computeLabExhaustive(b);
  EXHAUSTIVE_CACHE.set(b.id, res);
  return res;
}
