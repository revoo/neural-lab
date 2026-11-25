// authored-by: Claude Opus 5.5
/**
 * Lab: Spend the Step Budget. The learner chooses where the sampler stops on its way from pure
 * noise to clean data, and whether it reuses its noise estimate (DDIM) or swaps some for fresh
 * noise (DDPM). The denoiser is exact, so the only thing being tested is the step plan.
 * Every plan with every step count is tried by `exhaustive` (fast enough to run in a browser).
 */
import { LAB_LEVELS, labRun, subsets, type LabBriefId } from './diffusion';

export type Brief = { id: LabBriefId; title: string; story: string; maxTv: number };

export const BRIEFS: Brief[] = [
  {
    id: 'ring',
    title: 'Brief 1 · The ring',
    story: 'Eight equal blobs on a ring. Draw 2,000 samples with at most 10% of the probability out of place.',
    maxTv: 0.1,
  },
  {
    id: 'smiley',
    title: 'Brief 2 · The smiley',
    story: 'Two eyes and a smile: nine blobs, some close together. At most 12% out of place.',
    maxTv: 0.12,
  },
  {
    id: 'lopsided',
    title: 'Brief 3 · The rare blob',
    story: 'One big blob holds 90% of the data; a small one far away holds 10%. Give the rare blob its fair share: at most 6% out of place.',
    maxTv: 0.06,
  },
];

export const ETAS = [0, 1] as const;
export const MAX_STEPS = 6;

export type Plan = { levels: number[]; eta: number };
export const planSteps = (p: Plan) => p.levels.length + 1;

/** Every plan with a given number of steps (denoiser calls), both samplers, and how it scores. */
export function plansWithSteps(id: LabBriefId, steps: number, noiseSeed?: number) {
  const out: (Plan & { tv: number })[] = [];
  for (const eta of ETAS)
    for (const levels of subsets(steps - 1)) out.push({ levels, eta, tv: labRun(id, levels, eta, noiseSeed).judge.tv });
  return out;
}

/** The fewest steps with which some plan meets the brief, and how many plans of that size do. */
export function exhaustive(b: Brief, noiseSeed?: number) {
  const perSteps: { steps: number; total: number; passing: number; ddim: number; ddpm: number; best: number }[] = [];
  let fewest = -1;
  for (let steps = 1; steps <= MAX_STEPS; steps++) {
    const all = plansWithSteps(b.id, steps, noiseSeed);
    const pass = all.filter((p) => p.tv <= b.maxTv);
    perSteps.push({
      steps,
      total: all.length,
      passing: pass.length,
      ddim: pass.filter((p) => p.eta === 0).length,
      ddpm: pass.filter((p) => p.eta === 1).length,
      best: Math.min(...all.map((p) => p.tv)),
    });
    if (pass.length && fewest < 0) {
      fewest = steps;
      break;
    }
  }
  return { fewest, perSteps };
}

export const levelLabel = (a: number) => String(a);
export { LAB_LEVELS };

export function stepBudgetHint(id: LabBriefId, levels: number[], offShape?: number, rareShare?: number | null): string {
  if (offShape !== undefined && offShape > 0.5) {
    return 'Most samples land in the empty gaps between blobs: the final jump to clean data starts from too much noise, so the best guess averages neighbouring blobs together.';
  }
  if (id === 'lopsided' && rareShare !== null && rareShare !== undefined && rareShare < 0.07) {
    return `The rare blob gets only ${(rareShare * 100).toFixed(1)}% (fair share 10%). Samples must split towards the distant blob while noise is still high enough for them to wander that far.`;
  }
  if (levels.some((l) => l < 0.05) && id === 'ring') {
    return 'Check the forward-noise chart: at very small signal shares, almost none of the ring shape survives, so spending denoiser calls there buys almost nothing.';
  }
  if (id === 'ring') {
    return 'Where does the work happen? Below a small signal share the shape has nearly vanished, while the final jump to clean data is a single best guess that blurs if started too noisy.';
  }
  if (id === 'smiley') {
    return 'The eyes and smile blobs sit close together, so samples must separate neighbours at lower noise. Compare both samplers as you place stops where those decisions happen.';
  }
  return 'Watch the rare blob’s share. Samples have to commit to the rare blob early while noise is still wide, then sharpen onto both blobs without starving either one.';
}

