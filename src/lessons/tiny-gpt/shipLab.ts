// authored-by: Claude Opus 5.5
/**
 * "Ship the tiny GPT" (capstone lab): a fixed training budget, measured with the standard estimate
 * compute ≈ 6 × parameters × training tokens. The learner picks depth, width, heads, context and
 * learning rate; the number of steps is whatever the budget allows. Every design in the grid below
 * was trained by `search.run.ts` on two fixed seeds and is scored by its mean validation loss.
 */
import { gptParamCount, type GptConfig } from '../../lib/compute/gpt';
import type { RunSpec } from '../../lib/compute/gptRun';
import { VOCAB } from '../../lib/corpora/grimm';

export const LAYERS = [1, 2, 3] as const;
export const WIDTHS = [16, 32, 48] as const;
export const HEADS = [1, 4] as const;
export const CONTEXTS = [16, 64] as const;
export const LRS = [1e-3, 3e-3, 1e-2] as const;
export const BATCH = 16;
/** Training compute allowed, in floating-point operations (6 · N · tokens). */
export const BUDGET = 9e10;
export const SEEDS = [1, 2] as const;

export type Design = { layers: number; d: number; heads: number; T: number; lr: number };

export const designKey = (x: Design) => `L${x.layers}-d${x.d}-h${x.heads}-T${x.T}-lr${x.lr}`;
export const designCfg = (x: Design): GptConfig => ({ vocab: VOCAB, ctx: x.T, d: x.d, heads: x.heads, layers: x.layers });
export const designParams = (x: Design) => gptParamCount(designCfg(x));
/** Steps the budget allows: ⌊budget / (6 · N · B · T)⌋. */
export const designSteps = (x: Design) => Math.floor(BUDGET / (6 * designParams(x) * BATCH * x.T));

export function designSpec(x: Design, seed: number): RunSpec {
  const steps = designSteps(x);
  return {
    cfg: designCfg(x),
    T: x.T,
    batch: BATCH,
    steps,
    lr: x.lr,
    warmup: Math.min(50, Math.floor(steps / 10)),
    minLrFrac: 0.1,
    opt: { beta1: 0.9, beta2: 0.99, weightDecay: 0.1, clip: 1 },
    initSeed: seed,
    dataSeed: 1000 + seed,
    evalRows: 32,
  };
}

export function allDesigns(): Design[] {
  const out: Design[] = [];
  for (const layers of LAYERS) for (const d of WIDTHS) for (const heads of HEADS) for (const T of CONTEXTS) for (const lr of LRS) out.push({ layers, d, heads, T, lr });
  return out;
}
