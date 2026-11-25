// authored-by: Neon High

// ------------------------------------------------------------------ 1. Discrete Assistant Reply Space & KL-Regularised RLHF (Goodhart's Law)

export interface CandidateReply {
  id: string;
  label: string;
  snippet: string;
  kind: 'great' | 'good' | 'bland' | 'bad' | 'hack';
  /** Reference SFT policy probability π_ref(y) (sums to 1.0 across all 8 replies). */
  refProb: number;
  /** True human helpfulness R*(y) (hidden from the proxy reward model). */
  trueReward: number;
  /** Learned proxy reward model r̂(y) — mostly accurate, except for 2 Goodhart loopholes! */
  proxyReward: number;
}

export const CANDIDATE_REPLIES: CandidateReply[] = [
  {
    id: 'crisp',
    label: 'Crisp Rayleigh explanation',
    snippet: 'Short blue wavelengths scatter far more strongly off air molecules than red.',
    kind: 'great',
    refProb: 0.17,
    trueReward: 3.2,
    proxyReward: 3.1,
  },
  {
    id: 'analogy',
    label: 'Clear + prism analogy',
    snippet: 'Sunlight mixes every colour; nitrogen and oxygen scatter blue light across the sky.',
    kind: 'great',
    refProb: 0.15,
    trueReward: 3.0,
    proxyReward: 2.9,
  },
  {
    id: 'textbook',
    label: 'Solid textbook summary',
    snippet: 'Rayleigh scattering varies inversely with the fourth power of wavelength.',
    kind: 'good',
    refProb: 0.2,
    trueReward: 2.1,
    proxyReward: 2.2,
  },
  {
    id: 'brief',
    label: 'Terse one-liner',
    snippet: 'Air scatters blue light.',
    kind: 'bland',
    refProb: 0.22,
    trueReward: 0.8,
    proxyReward: 0.6,
  },
  {
    id: 'vague',
    label: 'Vague filler',
    snippet: 'The atmosphere interacts with sunlight in ways that make it appear blue.',
    kind: 'bland',
    refProb: 0.17,
    trueReward: -0.2,
    proxyReward: 0.1,
  },
  {
    id: 'wrong',
    label: 'Ocean-reflection myth',
    snippet: 'The sky is blue because it reflects the colour of the oceans.',
    kind: 'bad',
    refProb: 0.08,
    trueReward: -2.5,
    proxyReward: -2.4,
  },
  {
    id: 'fluff-hack',
    label: 'Sycophantic bullet-spam (hack #1)',
    snippet: 'What an extraordinary, brilliant question! Here are 12 enthusiastic points…',
    kind: 'hack',
    refProb: 0.008,
    trueReward: -1.4,
    proxyReward: 4.1,
  },
  {
    id: 'gibberish-hack',
    label: 'Reward-model glitch string (hack #2)',
    snippet: 'Certainly! Absolutely! Helpful accurate clear sky blue Rayleigh certainly!!',
    kind: 'hack',
    refProb: 0.002,
    trueReward: -2.8,
    proxyReward: 5.0,
  },
];

export interface PolicyMetrics {
  probs: number[];
  expectedTrue: number;
  expectedProxy: number;
  klNats: number; // D_KL(π || π_ref) = sum π(y) ln(π(y) / π_ref(y))
  hackProb: number; // total probability on 'hack' replies
  greatProb: number; // total probability on 'great' replies
}

export function evaluatePolicy(probs: number[], replies: CandidateReply[] = CANDIDATE_REPLIES): PolicyMetrics {
  let expectedTrue = 0;
  let expectedProxy = 0;
  let klNats = 0;
  let hackProb = 0;
  let greatProb = 0;
  for (let i = 0; i < replies.length; i++) {
    const p = probs[i];
    const r = replies[i];
    expectedTrue += p * r.trueReward;
    expectedProxy += p * r.proxyReward;
    if (p > 1e-15) klNats += p * Math.log(p / r.refProb);
    if (r.kind === 'hack') hackProb += p;
    if (r.kind === 'great') greatProb += p;
  }
  return { probs, expectedTrue, expectedProxy, klNats, hackProb, greatProb };
}

/**
 * Exact closed-form optimum of the KL-regularised RLHF objective:
 *   max_π  E_π[r̂(y)] - β · D_KL(π || π_ref)
 *   => π*(y) ∝ π_ref(y) · exp(r̂(y) / β)
 */
export function exactRlhfOptimum(beta: number, replies: CandidateReply[] = CANDIDATE_REPLIES): PolicyMetrics {
  const b = Math.max(0.02, beta);
  const logScores = replies.map((r) => Math.log(r.refProb) + r.proxyReward / b);
  const m = Math.max(...logScores);
  const exps = logScores.map((s) => Math.exp(s - m));
  const Z = exps.reduce((a, c) => a + c, 0);
  const probs = exps.map((e) => e / Z);
  return evaluatePolicy(probs, replies);
}

/** Sweeps β from 10.0 down to 0.08 to trace the classic Goodhart over-optimisation curve. */
export function traceGoodhartCurve(replies: CandidateReply[] = CANDIDATE_REPLIES) {
  const betas = [
    12, 8, 5, 3.5, 2.5, 1.8, 1.4, 1.1, 0.9, 0.75, 0.6, 0.5, 0.4, 0.32, 0.25, 0.2, 0.16, 0.12, 0.09, 0.07,
  ];
  const pts = betas.map((beta) => ({ beta, ...exactRlhfOptimum(beta, replies) }));
  let peak = pts[0];
  for (const p of pts) {
    if (p.expectedTrue > peak.expectedTrue) peak = p;
  }
  return { pts, peak };
}

// ------------------------------------------------------------------ 2. Direct Preference Optimisation (DPO) Loss & Gradients

export function sigmoid(x: number): number {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

export interface DpoPointMetrics {
  logRatioW: number; // ln(π(y_w) / π_ref(y_w))
  logRatioL: number; // ln(π(y_l) / π_ref(y_l))
  implicitRw: number; // β * logRatioW
  implicitRl: number; // β * logRatioL
  margin: number; // implicitRw - implicitRl
  loss: number; // -ln σ(margin)
  gradWeight: number; // σ(-margin): how hard DPO pushes this pair!
  winProb: number; // σ(margin): implicit Bradley-Terry P(y_w ≻ y_l)
}

export function evaluateDpoPoint(logRatioW: number, logRatioL: number, beta: number): DpoPointMetrics {
  const implicitRw = beta * logRatioW;
  const implicitRl = beta * logRatioL;
  const margin = implicitRw - implicitRl;
  // Numerically stable softplus(-margin) = ln(1 + exp(-margin))
  const loss = margin > 20 ? Math.exp(-margin) : margin < -20 ? -margin : Math.log1p(Math.exp(-margin));
  const gradWeight = sigmoid(-margin);
  const winProb = sigmoid(margin);
  return { logRatioW, logRatioL, implicitRw, implicitRl, margin, loss, gradWeight, winProb };
}

// ------------------------------------------------------------------ 3. Live DPO vs SFT Training on Tabular Softmax Policy

export interface PreferencePair {
  winner: number; // index into CANDIDATE_REPLIES
  loser: number; // index into CANDIDATE_REPLIES
  weight?: number;
}

export const CLEAN_PREF_PAIRS: PreferencePair[] = [
  { winner: 0, loser: 3 }, // Crisp > Terse
  { winner: 0, loser: 5 }, // Crisp > Myth
  { winner: 0, loser: 6 }, // Crisp > Sycophantic hack
  { winner: 0, loser: 7 }, // Crisp > Glitch hack
  { winner: 1, loser: 4 }, // Analogy > Vague
  { winner: 1, loser: 6 }, // Analogy > Sycophantic hack
  { winner: 2, loser: 5 }, // Textbook > Myth
  { winner: 2, loser: 7 }, // Textbook > Glitch hack
];

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/**
 * Exact average DPO loss & analytical gradient ∂L/∂logits for a tabular softmax policy
 * π_θ(y) = softmax(logits), starting from reference logits = ln(π_ref).
 * Verified against central finite differences in dpo.test.ts!
 */
export function dpoLossAndGrad(
  logits: number[],
  refLogits: number[],
  pairs: PreferencePair[],
  beta: number
): { loss: number; grad: number[] } {
  const probs = softmax(logits);
  const refProbs = softmax(refLogits);
  const K = logits.length;
  const grad = new Array<number>(K).fill(0);
  let totalLoss = 0;
  let totalWeight = 0;

  for (const pair of pairs) {
    const w = pair.weight ?? 1;
    totalWeight += w;
    const iW = pair.winner;
    const iL = pair.loser;

    // ln(π(y_w)/π_ref(y_w)) - ln(π(y_l)/π_ref(y_l))
    // Notice log Z cancels inside the difference of log-probabilities!
    // ln π(w) - ln π(l) = logits[w] - logits[l] !
    const logRatioW = Math.log(Math.max(1e-30, probs[iW])) - Math.log(Math.max(1e-30, refProbs[iW]));
    const logRatioL = Math.log(Math.max(1e-30, probs[iL])) - Math.log(Math.max(1e-30, refProbs[iL]));
    const margin = beta * (logRatioW - logRatioL);

    const pairLoss = margin > 20 ? Math.exp(-margin) : margin < -20 ? -margin : Math.log1p(Math.exp(-margin));
    totalLoss += w * pairLoss;

    // ∂L_pair / ∂margin = -σ(-margin)
    // ∂margin / ∂logits[k] = β * (1_{k=w} - 1_{k=l})   (because log Z cancels!)
    const gMargin = -w * sigmoid(-margin);
    grad[iW] += gMargin * beta;
    grad[iL] -= gMargin * beta;
  }

  const invW = 1 / Math.max(1e-12, totalWeight);
  for (let k = 0; k < K; k++) grad[k] *= invW;
  return { loss: totalLoss * invW, grad };
}

export interface LiveDpoRun {
  mode: 'sft' | 'dpo';
  beta: number;
  steps: number;
  history: Array<{ step: number; expectedTrue: number; klNats: number; greatProb: number; hackProb: number; loss: number }>;
  final: PolicyMetrics;
  finalLoss: number;
}

export function runLiveAlignment(params: {
  mode: 'sft' | 'dpo';
  beta: number;
  steps: number;
  lr?: number;
  pairs?: PreferencePair[];
}): LiveDpoRun {
  const { mode, beta, steps } = params;
  const pairs = params.pairs ?? CLEAN_PREF_PAIRS;
  const lr = params.lr ?? (mode === 'sft' ? 0.35 : 0.6);

  const refLogits = CANDIDATE_REPLIES.map((r) => Math.log(r.refProb));
  const logits = refLogits.slice();
  const history: LiveDpoRun['history'] = [];
  let lastLoss = 0;

  for (let s = 0; s <= steps; s++) {
    const probs = softmax(logits);
    const m = evaluatePolicy(probs);
    let loss = 0;
    const grad = new Array<number>(logits.length).fill(0);

    if (mode === 'dpo') {
      const res = dpoLossAndGrad(logits, refLogits, pairs, beta);
      loss = res.loss;
      for (let k = 0; k < grad.length; k++) grad[k] = res.grad[k];
    } else {
      // SFT-on-chosen-only: minimise -ln π(y_w) across winners (ignores losers completely!)
      let wSum = 0;
      for (const p of pairs) {
        const w = p.weight ?? 1;
        wSum += w;
        loss += -w * Math.log(Math.max(1e-30, probs[p.winner]));
        for (let k = 0; k < grad.length; k++) {
          grad[k] += w * (probs[k] - (k === p.winner ? 1 : 0));
        }
      }
      loss /= wSum;
      for (let k = 0; k < grad.length; k++) grad[k] /= wSum;
    }

    lastLoss = loss;
    if (s % Math.max(1, Math.floor(steps / 20)) === 0 || s === steps) {
      history.push({ step: s, expectedTrue: m.expectedTrue, klNats: m.klNats, greatProb: m.greatProb, hackProb: m.hackProb, loss });
    }
    if (s === steps) break;

    for (let k = 0; k < logits.length; k++) logits[k] -= lr * grad[k];
  }

  return {
    mode,
    beta,
    steps,
    history,
    final: evaluatePolicy(softmax(logits)),
    finalLoss: lastLoss,
  };
}

// ------------------------------------------------------------------ 4. Lab: Tune the Alignment Pipeline (3 briefs, exhaustive)

export const LAB_BETAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0] as const;
export const LAB_STEPS = [10, 25, 50, 100, 200] as const;
export const LAB_METHODS = ['rlhf-proxy', 'sft', 'dpo'] as const;
export type LabMethod = (typeof LAB_METHODS)[number];

export interface AlignmentLabConfig {
  method: LabMethod;
  beta: number;
  steps: number; // ignored for exact closed-form rlhf-proxy (uses exact β optimum)
}

export interface AlignmentBrief {
  id: string;
  title: string;
  scenario: string;
  allowDpo: boolean;
  noisyPairs?: PreferencePair[];
  minTrueReward: number;
  maxKlNats: number;
  maxHackProb: number;
  defaultConfig: AlignmentLabConfig;
}

// Noisy preference dataset (human raters sometimes fall for the sycophantic bullet-spam reply!)
export const NOISY_PREF_PAIRS: PreferencePair[] = [
  ...CLEAN_PREF_PAIRS,
  { winner: 6, loser: 2, weight: 2.5 }, // Fluff hack mislabelled > Textbook
  { winner: 6, loser: 3, weight: 2.5 }, // Fluff hack mislabelled > Terse
  { winner: 6, loser: 4, weight: 2.0 }, // Fluff hack mislabelled > Vague
];

export const ALIGNMENT_BRIEFS: AlignmentBrief[] = [
  {
    id: 'tame-rlhf',
    title: 'Brief 1 · Tame a hackable reward model',
    scenario:
      'You only have the proxy reward model (no DPO preference pipeline yet). Choose the KL penalty β to lift true human helpfulness ≥ 2.05 while keeping KL divergence ≤ 1.20 nats and loophole hack probability ≤ 15%.',
    allowDpo: false,
    minTrueReward: 2.05,
    maxKlNats: 1.2,
    maxHackProb: 0.15,
    defaultConfig: { method: 'rlhf-proxy', beta: 0.1, steps: 50 },
  },
  {
    id: 'clean-dpo',
    title: 'Brief 2 · Clean preference pairs (DPO vs SFT)',
    scenario:
      'Now you have clean human preference pairs (including explicit rejections of the sycophantic hacks). Push true human helpfulness ≥ 2.85 while keeping KL ≤ 1.60 nats and hack probability ≤ 0.5%.',
    allowDpo: true,
    minTrueReward: 2.85,
    maxKlNats: 1.6,
    maxHackProb: 0.005,
    defaultConfig: { method: 'sft', beta: 0.5, steps: 10 },
  },
  {
    id: 'noisy-dpo',
    title: 'Brief 3 · Noisy human raters (sycophancy bias)',
    scenario:
      'Some human raters fell for the sycophantic bullet-spam reply! Achieve true human helpfulness ≥ 2.40 while keeping KL ≤ 1.20 nats and hack probability ≤ 5% despite the noisy labels.',
    allowDpo: true,
    noisyPairs: NOISY_PREF_PAIRS,
    minTrueReward: 2.4,
    maxKlNats: 1.2,
    maxHackProb: 0.05,
    defaultConfig: { method: 'rlhf-proxy', beta: 0.25, steps: 50 },
  },
];

export interface AlignmentLabEval {
  metrics: PolicyMetrics;
  checks: {
    meetsTrue: boolean;
    meetsKl: boolean;
    meetsHack: boolean;
  };
  passes: boolean;
}

export function evaluateAlignmentBrief(brief: AlignmentBrief, cfg: AlignmentLabConfig): AlignmentLabEval {
  const effectiveMethod: LabMethod = brief.allowDpo ? cfg.method : 'rlhf-proxy';
  let metrics: PolicyMetrics;
  if (effectiveMethod === 'rlhf-proxy') {
    metrics = exactRlhfOptimum(cfg.beta);
  } else {
    const run = runLiveAlignment({
      mode: effectiveMethod,
      beta: cfg.beta,
      steps: cfg.steps,
      pairs: brief.noisyPairs ?? CLEAN_PREF_PAIRS,
    });
    metrics = run.final;
  }

  const checks = {
    meetsTrue: metrics.expectedTrue >= brief.minTrueReward,
    meetsKl: metrics.klNats <= brief.maxKlNats,
    meetsHack: metrics.hackProb <= brief.maxHackProb,
  };
  return {
    metrics,
    checks,
    passes: checks.meetsTrue && checks.meetsKl && checks.meetsHack,
  };
}

export function solveAlignmentBrief(brief: AlignmentBrief): {
  totalConfigs: number;
  passingCount: number;
  bestConfig: AlignmentLabConfig;
  bestEval: AlignmentLabEval;
} {
  const methods: LabMethod[] = brief.allowDpo ? ['rlhf-proxy', 'sft', 'dpo'] : ['rlhf-proxy'];
  let totalConfigs = 0;
  let passingCount = 0;
  let bestConfig: AlignmentLabConfig | null = null;
  let bestEval: AlignmentLabEval | null = null;

  for (const method of methods) {
    const betas = method === 'sft' ? [0.5] : [...LAB_BETAS];
    const stepsList = method === 'rlhf-proxy' ? [50] : [...LAB_STEPS];
    for (const beta of betas) {
      for (const steps of stepsList) {
        totalConfigs++;
        const cfg: AlignmentLabConfig = { method, beta, steps };
        const ev = evaluateAlignmentBrief(brief, cfg);
        if (ev.passes) {
          passingCount++;
          if (!bestEval || ev.metrics.expectedTrue > bestEval.metrics.expectedTrue + 1e-9) {
            bestConfig = cfg;
            bestEval = ev;
          }
        }
      }
    }
  }

  return {
    totalConfigs,
    passingCount,
    bestConfig: bestConfig!,
    bestEval: bestEval!,
  };
}

/**
 * Hint for an unsolved brief. It points at the failing check and only names a
 * method the brief unlocks (pinned by a test: Brief 1 never mentions DPO).
 */
export function alignmentHint(brief: AlignmentBrief, cfg: AlignmentLabConfig, ev: AlignmentLabEval): string {
  const method: LabMethod = brief.allowDpo ? cfg.method : 'rlhf-proxy';
  if (method === 'sft') {
    return 'SFT only imitates the chosen replies: it never sees which replies people rejected. Compare its hack bar with the other methods.';
  }
  if (!ev.checks.meetsHack) {
    const base = 'The hack bar is too tall: the policy is chasing the loophole replies the reward model overrates. Which way does β move the leash back to the reference?';
    return method === 'rlhf-proxy' && brief.allowDpo
      ? `${base} Or try a method that learns from explicit rejections instead of the proxy score.`
      : base;
  }
  if (!ev.checks.meetsKl) {
    return method === 'rlhf-proxy'
      ? 'KL is too high: the policy has wandered too far from the reference. A larger β pulls it back.'
      : 'KL is too high: the policy has wandered too far from the reference. A larger β or fewer steps pulls it back.';
  }
  return method === 'rlhf-proxy'
    ? 'Safe but timid: true helpfulness is too low because the leash is tight. Loosen β one notch at a time and watch the hack bar.'
    : 'Safe but timid: true helpfulness is too low. Loosen β a little, or train for more steps.';
}
