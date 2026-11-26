// authored-by: Claude Opus 5.5
/**
 * Lab: Spend the Test-Time Budget. Every number is exact (closed forms in `reasoning.ts`), so each
 * brief's solutions and optimum are found by enumerating its whole grid (pinned in `reasoning.test.ts`).
 */
import { benchAccuracy, BENCH, flops, samples, tokensPerSample, type Config, type Strategy, type Tier, type Verifier } from './reasoning';

export const LAB_TIERS: Tier[] = ['small', 'medium', 'large'];
export const LAB_STRATEGIES: Strategy[] = ['single', 'majority', 'best'];
export const LAB_N = [2, 4, 8, 16, 32, 64];
export const LAB_VERIFIERS: Verifier[] = ['none', 'noisy', 'exact'];
export const LAB_THINKING = [0, 100, 200, 300, 400];

export const HARD = BENCH.filter((p) => p.steps >= 7);

/** The reference for Brief 1: the large model answering once after a short think. */
export const LARGE_REF: Config = { tier: 'large', strategy: 'single', n: 1, verifier: 'none', thinking: 100 };
export const B2_BAR = 0.8;
export const B3_BAR = 0.85;
export const B3_TOKENS = 3000;

export type BriefId = 'cheap' | 'noisy' | 'hard';
export const BRIEFS: { id: BriefId; tab: string }[] = [
  { id: 'cheap', tab: '1 · No checker' },
  { id: 'noisy', tab: '2 · Noisy judge' },
  { id: 'hard', tab: '3 · Hard problems' },
];

/** What each brief fixes; the learner sets the rest. */
export const LOCKS: Record<BriefId, Partial<Config> & { tiers?: Tier[]; strategies?: Strategy[] }> = {
  cheap: { verifier: 'none', tiers: ['small', 'medium'], strategies: ['single', 'majority'] },
  noisy: { tier: 'medium', strategy: 'best', verifier: 'noisy' },
  hard: { tier: 'medium', strategy: 'best', verifier: 'exact' },
};

export function applyLocks(brief: BriefId, c: Config): Config {
  const L = LOCKS[brief];
  const out = { ...c, ...(L.tier ? { tier: L.tier } : {}), ...(L.strategy ? { strategy: L.strategy } : {}), ...(L.verifier ? { verifier: L.verifier } : {}) };
  if (L.tiers && !L.tiers.includes(out.tier)) out.tier = L.tiers[0];
  if (L.strategies && !L.strategies.includes(out.strategy)) out.strategy = L.strategies[0];
  if (out.strategy === 'single') out.n = 1;
  else if (out.n < 2) out.n = 2;
  return out;
}

export const generatedTokens = (c: Config) => samples(c) * tokensPerSample(c.thinking);

export type Score = { acc: number; flops: number; tokens: number };
export function score(brief: BriefId, c: Config): Score {
  return { acc: benchAccuracy(c, brief === 'hard' ? HARD : BENCH), flops: flops(c), tokens: generatedTokens(c) };
}

export const refScore = () => score('cheap', LARGE_REF);

export function passes(brief: BriefId, s: Score) {
  if (brief === 'cheap') {
    const ref = refScore();
    return s.acc >= ref.acc && s.flops < ref.flops;
  }
  if (brief === 'noisy') return s.acc >= B2_BAR;
  return s.acc >= B3_BAR && s.tokens <= B3_TOKENS;
}

/** Every distinct configuration a brief allows. */
export function briefGrid(brief: BriefId): Config[] {
  const out: Config[] = [];
  const seen = new Set<string>();
  for (const tier of LAB_TIERS)
    for (const strategy of LAB_STRATEGIES)
      for (const n of strategy === 'single' ? [1] : LAB_N)
        for (const verifier of LAB_VERIFIERS)
          for (const thinking of LAB_THINKING) {
            const c = applyLocks(brief, { tier, strategy, n, verifier, thinking });
            // a verifier only matters to best-of-N
            if (c.strategy !== 'best') c.verifier = LOCKS[brief].verifier ?? 'none';
            const k = JSON.stringify(c);
            if (!seen.has(k)) {
              seen.add(k);
              out.push(c);
            }
          }
  return out;
}

/** Evidence-only hints: they describe the current numbers, never a setting to choose. */
export function labHint(brief: BriefId, c: Config, s: Score): string {
  const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
  if (brief === 'cheap') {
    const ref = refScore();
    if (s.acc < ref.acc)
      return `${pct(s.acc)} against the large model’s ${pct(ref.acc)}. Look at where the misses are: a model that must skip written steps pays for each skipped step four times over, and a vote can only help when the right answer is the most common one.`;
    return `Accurate enough, but this costs ${(s.flops / 1e12).toFixed(1)} TFLOP per problem against the large model’s ${(ref.flops / 1e12).toFixed(1)}. Every sample pays for all of its tokens, thinking included.`;
  }
  if (brief === 'noisy')
    return `${pct(s.acc)} with ${samples(c)} samples per problem. Each extra sample is another chance to find a right answer, and another chance to find a wrong one the judge happens to love. Compare a few neighbouring sample counts.`;
  if (s.tokens > B3_TOKENS) return `This spends ${s.tokens.toLocaleString()} generated tokens per problem, over the ${B3_TOKENS.toLocaleString()} budget.`;
  return `${pct(s.acc)} on the hard problems for ${s.tokens.toLocaleString()} tokens. The checker only needs one correct attempt, but on a 7- to 10-step problem an attempt that skips steps almost never is one.`;
}
