// authored-by: Neon High
/**
 * Parametric neural scaling laws:
 *   L(N, D) = E + A / N^α + B / D^β,   with training compute C ≈ 6 N D
 *
 * Uses the Besiroglu et al. (2024) replication refit of Hoffmann et al.'s (2022) Chinchilla runs
 * (E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658), which faithfully reproduces
 * the ~20 tokens/parameter compute-optimal rule across frontier scales (unlike the rounded
 * Approach-3 constants printed in the 2022 paper, which we also export for honest comparison).
 */

export const CHINCHILLA = {
  E: 1.8172,
  A: 482.01,
  B: 2085.43,
  alpha: 0.3478,
  beta: 0.3658,
} as const;

/** The rounded Approach-3 constants printed in Hoffmann et al. (2022), shown in the Deeper honesty note. */
export const HOFFMANN_ROUNDED = {
  E: 1.69,
  A: 406.4,
  B: 410.7,
  alpha: 0.34,
  beta: 0.28,
} as const;

/** Effective sustained BF16 FLOP/s for one H100 GPU at ~30% MFU (989 TFLOP/s peak × 0.30 ≈ 300 TFLOP/s). */
export const H100_EFFECTIVE_FLOPS = 3e14;
export const SECONDS_PER_DAY = 86400;

export function trainFlops(N: number, D: number) {
  return 6 * N * D;
}

export function inferenceFlops(N: number, dInf: number) {
  return 2 * N * dInf;
}

export function h100Days(flops: number) {
  return flops / (H100_EFFECTIVE_FLOPS * SECONDS_PER_DAY);
}

export function lossTerms(N: number, D: number, fit: typeof CHINCHILLA | typeof HOFFMANN_ROUNDED = CHINCHILLA) {
  const irreducible = fit.E;
  const paramPenalty = fit.A / N ** fit.alpha;
  const dataPenalty = fit.B / D ** fit.beta;
  const total = irreducible + paramPenalty + dataPenalty;
  return { irreducible, paramPenalty, dataPenalty, total, excess: total - irreducible, ppl: Math.exp(total) };
}

export function predictedLoss(N: number, D: number, fit: typeof CHINCHILLA | typeof HOFFMANN_ROUNDED = CHINCHILLA) {
  return lossTerms(N, D, fit).total;
}

/** Exact calculus optimum of L(N, D) subject to 6 N D = C. */
export function chinchillaOptimal(C: number, fit: typeof CHINCHILLA | typeof HOFFMANN_ROUNDED = CHINCHILLA) {
  const a = fit.beta / (fit.alpha + fit.beta);
  const b = fit.alpha / (fit.alpha + fit.beta);
  const G = ((fit.alpha * fit.A) / (fit.beta * fit.B)) ** (1 / (fit.alpha + fit.beta));
  const N = G * (C / 6) ** a;
  const D = (C / 6) ** b / G;
  const terms = lossTerms(N, D, fit);
  return { N, D, ratio: D / N, ...terms, a, b };
}

/** Training tokens D needed for a model of size N to reach target loss L (or null if N is too small even with infinite data). */
export function tokensForLoss(N: number, targetLoss: number, fit: typeof CHINCHILLA | typeof HOFFMANN_ROUNDED = CHINCHILLA): number | null {
  const rem = targetLoss - fit.E - fit.A / N ** fit.alpha;
  if (rem <= 0) return null;
  return (fit.B / rem) ** (1 / fit.beta);
}

/**
 * Compute lifecycle-optimal (N, D_train) to achieve `targetLoss` while serving `dInf` lifetime
 * inference tokens: minimises C_total = 6 N D_train(N) + 2 N dInf.
 * Evaluated over a dense logarithmic grid (0.001 dex resolution, ~2,000 points).
 */
export function lifecycleOptimal(targetLoss: number, dInf: number) {
  let best = { N: 0, D: 0, trainC: Infinity, infC: Infinity, totalC: Infinity, ratio: 0 };
  for (let logN = 9.0; logN <= 11.5; logN += 0.001) {
    const N = 10 ** logN;
    const D = tokensForLoss(N, targetLoss);
    if (D === null) continue;
    const trainC = trainFlops(N, D);
    const infC = inferenceFlops(N, dInf);
    const totalC = trainC + infC;
    if (totalC < best.totalC) {
      best = { N, D, trainC, infC, totalC, ratio: D / N };
    }
  }
  return best;
}

export type FrontierRun = {
  id: string;
  name: string;
  year: number;
  N: number;
  D: number;
  note: string;
};

export const FRONTIER_RUNS: FrontierRun[] = [
  { id: 'gpt3', name: 'GPT-3 175B', year: 2020, N: 175e9, D: 300e9, note: 'Under-trained (1.7 tok/param) under pre-Chinchilla Kaplan assumptions.' },
  { id: 'gopher', name: 'Gopher 280B', year: 2021, N: 280e9, D: 300e9, note: 'Severely under-trained (1.1 tok/param); beaten by Chinchilla at 4× smaller size.' },
  { id: 'chinchilla', name: 'Chinchilla 70B', year: 2022, N: 70e9, D: 1.4e12, note: 'Right at the training-compute optimum (~20 tok/param).' },
  { id: 'llama3-8b', name: 'Llama 3 8B', year: 2024, N: 8e9, D: 15e12, note: 'Deliberately over-trained (1,875 tok/param) for cheap inference at scale.' },
  { id: 'llama3-70b', name: 'Llama 3 70B', year: 2024, N: 70e9, D: 15e12, note: '10× more tokens than Chinchilla-optimal (214 tok/param), far cheaper to serve.' },
];

export type BudgetBrief = {
  id: string;
  name: string;
  brief: string;
  maxTrainC?: number;
  maxN?: number;
  dInf?: number;
  targetLoss: number;
  maxTotalC?: number;
  check: (N: number, D: number) => boolean;
};

export const BUDGET_BRIEFS: BudgetBrief[] = [
  {
    id: 'research',
    name: '1. Research Run',
    brief: 'Fixed training budget C ≤ 5.0 × 10²³ FLOPs (Gopher scale). Reach predicted loss L ≤ 1.985 (Gopher 280B only got 2.000).',
    maxTrainC: 5.0e23,
    targetLoss: 1.985,
    check: (N, D) => trainFlops(N, D) <= 5.0e23 * 1.001 && predictedLoss(N, D) <= 1.985,
  },
  {
    id: 'edge',
    name: '2. Single-GPU Edge Model',
    brief: 'Must fit on one 24 GB GPU at 4-bit quantisation (N ≤ 20B params) with C ≤ 6.0 × 10²³ FLOPs. Reach predicted loss L ≤ 1.998.',
    maxTrainC: 6.0e23,
    maxN: 20e9,
    targetLoss: 1.998,
    check: (N, D) => N <= 20e9 * 1.001 && trainFlops(N, D) <= 6.0e23 * 1.001 && predictedLoss(N, D) <= 1.998,
  },
  {
    id: 'product',
    name: '3. Billion-User Chatbot',
    brief: 'Hit quality L ≤ 2.00 while serving 20 trillion lifetime inference tokens (2 N D_inf), keeping TOTAL lifecycle compute (train + inference) ≤ 1.25 × 10²⁴ FLOPs.',
    dInf: 20e12,
    targetLoss: 2.0,
    maxTotalC: 1.25e24,
    check: (N, D) => predictedLoss(N, D) <= 2.0 && trainFlops(N, D) + inferenceFlops(N, 20e12) <= 1.25e24,
  },
];

export function fmtCount(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return n.toLocaleString('en-GB');
}

export function fmtFlops(c: number): string {
  if (c === 0) return '0';
  const exp = Math.floor(Math.log10(c));
  const mant = c / 10 ** exp;
  return `${mant.toFixed(2)} × 10^${exp}`;
}
