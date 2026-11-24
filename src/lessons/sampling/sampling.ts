/**
 * Decoding & sampling engine: temperature scaling, top-k, top-p (nucleus) filtering,
 * sharp vs flat context benchmarks, exhaustive slider-grid search for the Sampler Lab,
 * and a 3-step beam-search vs greedy decision tree.
 */

export type TokenOption = { word: string; logit: number; valid: boolean };
export type SamplingContext = {
  id: 'sharp' | 'medium' | 'flat';
  label: string;
  prompt: string;
  tokens: TokenOption[];
};

export const CONTEXTS: SamplingContext[] = [
  {
    id: 'sharp',
    label: 'Sharp (factual)',
    prompt: 'The capital of France is',
    tokens: [
      { word: 'Paris', logit: 6.2, valid: true },
      { word: 'Lyon', logit: 2.3, valid: false },
      { word: 'France', logit: 2.0, valid: false },
      { word: 'London', logit: 1.7, valid: false },
      { word: 'located', logit: 1.6, valid: false },
      { word: 'known', logit: 1.4, valid: false },
      { word: 'one', logit: 1.2, valid: false },
      { word: 'a', logit: 1.1, valid: false },
      { word: 'the', logit: 1.0, valid: false },
      { word: 'very', logit: 0.9, valid: false },
      { word: 'famous', logit: 0.8, valid: false },
      { word: 'banana', logit: 0.5, valid: false },
      { word: 'purple', logit: 0.4, valid: false },
      { word: '742', logit: 0.3, valid: false },
      { word: 'keyboard', logit: 0.2, valid: false },
      { word: 'bicycle', logit: 0.1, valid: false },
    ],
  },
  {
    id: 'medium',
    label: 'Medium (everyday)',
    prompt: 'The cat sat on the',
    tokens: [
      { word: 'mat', logit: 4.1, valid: true },
      { word: 'warm', logit: 3.6, valid: true },
      { word: 'rug', logit: 3.4, valid: true },
      { word: 'couch', logit: 3.1, valid: true },
      { word: 'floor', logit: 2.8, valid: true },
      { word: 'roof', logit: 2.1, valid: true },
      { word: 'chair', logit: 1.9, valid: true },
      { word: 'window', logit: 1.5, valid: true },
      { word: 'table', logit: 1.2, valid: true },
      { word: 'porch', logit: 0.8, valid: true },
      { word: 'cloud', logit: 0.2, valid: false },
      { word: 'equation', logit: 0.0, valid: false },
      { word: 'democracy', logit: -0.2, valid: false },
      { word: '742', logit: -0.4, valid: false },
      { word: 'volcano', logit: -0.6, valid: false },
      { word: 'trombone', logit: -0.8, valid: false },
    ],
  },
  {
    id: 'flat',
    label: 'Flat (open-ended)',
    prompt: 'I opened the box and found a',
    tokens: [
      { word: 'small', logit: 3.1, valid: true },
      { word: 'key', logit: 3.0, valid: true },
      { word: 'letter', logit: 2.95, valid: true },
      { word: 'book', logit: 2.85, valid: true },
      { word: 'note', logit: 2.8, valid: true },
      { word: 'wooden', logit: 2.7, valid: true },
      { word: 'silver', logit: 2.65, valid: true },
      { word: 'tiny', logit: 2.55, valid: true },
      { word: 'photo', logit: 2.45, valid: true },
      { word: 'map', logit: 2.35, valid: true },
      { word: 'watch', logit: 2.25, valid: true },
      { word: 'ring', logit: 2.1, valid: true },
      { word: 'galaxy', logit: 0.4, valid: false },
      { word: 'theorem', logit: 0.2, valid: false },
      { word: '742', logit: 0.0, valid: false },
      { word: 'trombone', logit: -0.2, valid: false },
    ],
  },
];

export type SampleResult = {
  /** Probabilities right after softmax(z / T), before any truncation. */
  raw: number[];
  /** Final re-normalised sampling probabilities after top-k and top-p. */
  probs: number[];
  /** Whether each token survived both top-k and top-p. */
  kept: boolean[];
  /** Number of surviving tokens. */
  poolSize: number;
  /** Shannon entropy H in nats of the final sampling distribution. */
  entropy: number;
  /** Effective number of equally likely choices: exp(H). */
  effectiveChoices: number;
  /** Total final probability mass placed on valid tokens. */
  validMass: number;
  /** Total final probability mass placed on off-topic / tail-junk tokens. */
  junkMass: number;
};

/** Apply temperature scaling → softmax → top-k → top-p (nucleus) → re-normalise. */
export function applySampling(tokens: TokenOption[], temperature: number, topK: number, topP: number): SampleResult {
  const t = Math.max(0.01, temperature);
  const scaled = tokens.map((x) => x.logit / t);
  const maxL = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  const raw = exps.map((e) => e / sum);

  const order = tokens.map((_, i) => i).sort((a, b) => raw[b] - raw[a]);
  const kept = new Array<boolean>(tokens.length).fill(false);

  let cum = 0;
  for (let rank = 0; rank < order.length; rank++) {
    if (rank >= topK) break;
    const idx = order[rank];
    kept[idx] = true;
    cum += raw[idx];
    if (cum >= topP) break;
  }

  let keptSum = 0;
  for (let i = 0; i < tokens.length; i++) if (kept[i]) keptSum += raw[i];
  const probs = raw.map((p, i) => (kept[i] ? p / keptSum : 0));

  let entropy = 0;
  let validMass = 0;
  let junkMass = 0;
  let poolSize = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (kept[i]) poolSize++;
    if (probs[i] > 0) {
      entropy -= probs[i] * Math.log(probs[i]);
      if (tokens[i].valid) validMass += probs[i];
      else junkMass += probs[i];
    }
  }
  return { raw, probs, kept, poolSize, entropy, effectiveChoices: Math.exp(entropy), validMass, junkMass };
}

// ------------------------------------------------------------------ Lab evaluation & exhaustive grid search

export const LAB_TEMPS = Array.from({ length: 20 }, (_, i) => Number(((i + 1) * 0.1).toFixed(1))); // 0.1..2.0
export const LAB_KS = Array.from({ length: 16 }, (_, i) => i + 1); // 1..16
export const LAB_PS = Array.from({ length: 19 }, (_, i) => Number((0.1 + i * 0.05).toFixed(2))); // 0.10..1.00

export type SamplerMetrics = {
  sharpAcc: number;
  mediumJunk: number;
  flatJunk: number;
  maxTailJunk: number;
  flatEff: number;
  sharpPool: number;
  flatPool: number;
};

export function evaluateSampler(temperature: number, topK: number, topP: number): SamplerMetrics {
  const s = applySampling(CONTEXTS[0].tokens, temperature, topK, topP);
  const m = applySampling(CONTEXTS[1].tokens, temperature, topK, topP);
  const f = applySampling(CONTEXTS[2].tokens, temperature, topK, topP);
  return {
    sharpAcc: s.validMass,
    mediumJunk: m.junkMass,
    flatJunk: f.junkMass,
    maxTailJunk: Math.max(m.junkMass, f.junkMass),
    flatEff: f.effectiveChoices,
    sharpPool: s.poolSize,
    flatPool: f.poolSize,
  };
}

export type LabLevel = {
  id: string;
  name: string;
  brief: string;
  /** Which sliders are unlocked (to let the learner feel why temperature-only and top-k-only hit walls). */
  lockTemp?: number;
  lockK?: number;
  lockP?: number;
  check: (m: SamplerMetrics) => boolean;
  /** Optional optimisation score to maximise (higher is better), when feasible. */
  score?: (m: SamplerMetrics) => number;
  /** Whether the level is provably impossible over its allowed grid. */
  impossible?: boolean;
};

export const LAB_LEVELS: LabLevel[] = [
  {
    id: 'factual',
    name: '1. Factual Q&A',
    brief: 'Hit ≥ 99% accuracy on the sharp prompt AND 0% tail junk on all prompts.',
    check: (m) => m.sharpAcc >= 0.99 && m.maxTailJunk === 0,
    score: (m) => m.flatEff,
  },
  {
    id: 'topk-trap',
    name: '2. The Top-k Trap',
    brief: 'Using ONLY top-k (T = 1.0, top-p = 1.0 locked), hit ≥ 98% sharp accuracy AND ≥ 9.0 effective choices on the flat prompt. Can it be done?',
    lockTemp: 1.0,
    lockP: 1.0,
    check: (m) => m.sharpAcc >= 0.98 && m.flatEff >= 9.0,
    impossible: true,
  },
  {
    id: 'balanced',
    name: '3. One Sampler for All',
    brief: 'All three knobs unlocked: hit ≥ 98% sharp accuracy, ≥ 9.5 effective choices on the flat prompt, AND 0.0% tail junk on medium & flat prompts.',
    check: (m) => m.sharpAcc >= 0.98 && m.flatEff >= 9.5 && m.maxTailJunk === 0,
    score: (m) => m.flatEff,
  },
];

export type LevelSolutionStats = {
  gridSize: number;
  passing: number;
  tempOnlyPassing: number;
  topKOnlyPassing: number;
  topPOnlyPassing: number;
  best: { T: number; k: number; p: number; metrics: SamplerMetrics } | null;
};

/** Exhaustively enumerate every valid setting on the discrete slider grid for a lab level. */
export function solveLabLevel(level: LabLevel): LevelSolutionStats {
  const ts = level.lockTemp !== undefined ? [level.lockTemp] : LAB_TEMPS;
  const ks = level.lockK !== undefined ? [level.lockK] : LAB_KS;
  const ps = level.lockP !== undefined ? [level.lockP] : LAB_PS;

  let gridSize = 0;
  let passing = 0;
  let tempOnlyPassing = 0;
  let topKOnlyPassing = 0;
  let topPOnlyPassing = 0;
  let best: { T: number; k: number; p: number; metrics: SamplerMetrics } | null = null;
  let bestScore = -Infinity;

  for (const T of ts) {
    for (const k of ks) {
      for (const p of ps) {
        gridSize++;
        const m = evaluateSampler(T, k, p);
        if (level.check(m)) {
          passing++;
          if (k === 16 && p === 1.0) tempOnlyPassing++;
          if (T === 1.0 && p === 1.0) topKOnlyPassing++;
          if (T === 1.0 && k === 16) topPOnlyPassing++;
          const sc = level.score ? level.score(m) : m.flatEff;
          if (sc > bestScore) {
            bestScore = sc;
            best = { T, k, p, metrics: m };
          }
        }
      }
    }
  }
  return { gridSize, passing, tempOnlyPassing, topKOnlyPassing, topPOnlyPassing, best };
}

// ------------------------------------------------------------------ Beam search vs greedy tree

export type BeamNode = { word: string; prob: number; children?: BeamNode[] };

export const BEAM_TREE: BeamNode[] = [
  {
    word: 'The',
    prob: 0.55,
    children: [
      {
        word: 'cat',
        prob: 0.4,
        children: [
          { word: 'slept', prob: 0.4 },
          { word: 'ran', prob: 0.35 },
        ],
      },
      {
        word: 'dog',
        prob: 0.35,
        children: [
          { word: 'barked', prob: 0.5 },
          { word: 'sat', prob: 0.3 },
        ],
      },
    ],
  },
  {
    word: 'A',
    prob: 0.45,
    children: [
      {
        word: 'bright',
        prob: 0.85,
        children: [
          { word: 'sun', prob: 0.9 },
          { word: 'moon', prob: 0.1 },
        ],
      },
      {
        word: 'cold',
        prob: 0.15,
        children: [
          { word: 'wind', prob: 0.6 },
          { word: 'rain', prob: 0.4 },
        ],
      },
    ],
  },
];

export type BeamPath = { words: string[]; prob: number; logProb: number };

/** Run beam search of width `beamWidth` for `steps` (1..3) on BEAM_TREE. */
export function runBeamSearch(beamWidth: number, steps = 3): BeamPath[] {
  let beam: { words: string[]; prob: number; node?: BeamNode }[] = [{ words: [], prob: 1 }];
  for (let s = 0; s < steps; s++) {
    const next: typeof beam = [];
    for (const b of beam) {
      const kids = s === 0 ? BEAM_TREE : b.node?.children ?? [];
      for (const k of kids) {
        next.push({ words: [...b.words, k.word], prob: b.prob * k.prob, node: k });
      }
    }
    next.sort((a, b) => b.prob - a.prob);
    beam = next.slice(0, beamWidth);
  }
  return beam.map(({ words, prob }) => ({ words, prob, logProb: Math.log(prob) }));
}
