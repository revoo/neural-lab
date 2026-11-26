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
/** Brief 2 and 3's target validation loss (nats per character, mean of both seeds). PROVISIONAL until the grid is complete. */
export const TARGET = 1.85;

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

// ------------------------------------------------------------------ scoring (against the recorded search)

export type Recorded = { params: number; steps: number; val: readonly number[]; curve: readonly (readonly (readonly [number, number])[])[] };

/** Mean validation loss over the seeds. */
export const meanVal = (r: Recorded) => r.val.reduce((s, v) => s + v, 0) / r.val.length;

export type BriefId = 'beat-bigram' | 'target' | 'lean';
export const BIGRAM = 2.286; // ngramLoss(train, val, 2), pinned in grimm.test.ts

export function passes(brief: BriefId, r: Recorded, target: number): boolean {
  const v = meanVal(r);
  return brief === 'beat-bigram' ? v < BIGRAM : v <= target;
}

/** Exhaustive over the recorded grid: the designs that pass, and the best by a criterion. */
export function gridSummary(results: Record<string, Recorded>, target: number) {
  const all = allDesigns().map((x) => ({ x, r: results[designKey(x)] })).filter((e) => e.r);
  const best = [...all].sort((a, b) => meanVal(a.r) - meanVal(b.r))[0];
  const beatBigram = all.filter((e) => passes('beat-bigram', e.r, target));
  const hitTarget = all.filter((e) => passes('target', e.r, target));
  const leanest = [...hitTarget].sort((a, b) => a.r.params - b.r.params || meanVal(a.r) - meanVal(b.r))[0];
  return { total: all.length, best, beatBigram: beatBigram.length, hitTarget: hitTarget.length, leanest, leanestCount: hitTarget.filter((e) => e.r.params === leanest?.r.params).length };
}

/**
 * Evidence about a finished design, never a setting to choose: how far the loss was still falling
 * at the end, how many steps the budget allowed, and how the two seeds compare.
 */
export function shipHint(x: Design, r: Recorded): string {
  const c = r.curve[0];
  const end = c[c.length - 1];
  const mid = c[Math.max(0, c.length - 4)];
  const stillFalling = mid[1] - end[1];
  const parts: string[] = [];
  parts.push(`${designParams(x).toLocaleString('en-GB')} parameters left room for ${r.steps.toLocaleString('en-GB')} steps of ${BATCH} × ${x.T} characters.`);
  if (stillFalling > 0.1) parts.push(`Over its last few checkpoints (from step ${mid[0].toLocaleString('en-GB')}) the loss was still falling by ${stillFalling.toFixed(2)}: it ran out of budget while still learning.`);
  else parts.push(`Over its last few checkpoints (from step ${mid[0].toLocaleString('en-GB')}) the loss moved by only ${stillFalling.toFixed(2)}: it had nearly stopped improving.`);
  const early = c.find(([s]) => s >= Math.min(20, r.steps));
  if (early && early[1] > c[0][1] - 0.05) parts.push(`By step ${early[0]} it had barely moved from where it started (${early[1].toFixed(2)}): its first steps were tiny.`);
  return parts.join(' ');
}
