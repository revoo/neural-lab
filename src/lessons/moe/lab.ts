// authored-by: Claude Opus 5.5
/**
 * Lab: Balance the Experts. The learner sets top-k, the load-balancing weight α, the capacity factor C
 * and the router noise of a 4-expert layer; every setting is trained on the same 5 seeds and scored on
 * the averages. The whole grid (2 × 5 × 4 × 3 = 120 settings) is searched by `search.run.ts` into
 * `results.ts` and re-verified in `moe.test.ts`.
 */
import { DEFAULTS, train, type MoeConfig } from './moe';

export const LAB_SEEDS = [1, 2, 3, 4, 5] as const;
export const LAB_K = [1, 2] as const;
export const LAB_ALPHA = [0, 0.01, 0.03, 0.1, 0.3] as const;
export const LAB_CAPACITY = [1, 1.25, 1.5, 2] as const;
export const LAB_NOISE = [0, 0.5, 1] as const;

export type LabSetting = { k: (typeof LAB_K)[number]; alpha: number; capacity: number; noise: number };

export const ACC_BAR = 0.95;
export const MAX_LOAD_BAR = 0.35;
export const DROP_BAR = 0.03;
export const TIGHT_C = 1.25;

export type LabScore = LabSetting & {
  /** Mean test accuracy over the 5 seeds. */
  acc: number;
  /** Worst seed's busiest-expert share. */
  maxLoad: number;
  /** Quietest expert's share on the worst seed. */
  minLoad: number;
  /** Mean share of token slots dropped (last 100 steps). */
  drop: number;
  /** Expert slots reserved per token: k × C (what the hardware pays for, used or not). */
  cost: number;
  perSeed: { acc: number; load: number[]; drop: number }[];
};

export const configOf = (s: LabSetting): MoeConfig => ({ ...DEFAULTS, k: s.k, alpha: s.alpha, capacity: s.capacity, noise: s.noise });

export function scoreSetting(s: LabSetting): LabScore {
  const runs = LAB_SEEDS.map((seed) => train(configOf(s), seed));
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  return {
    ...s,
    acc: mean(runs.map((r) => r.acc)),
    maxLoad: Math.max(...runs.map((r) => r.maxLoad)),
    minLoad: Math.min(...runs.map((r) => Math.min(...r.load))),
    drop: mean(runs.map((r) => r.dropRate)),
    cost: s.k * s.capacity,
    perSeed: runs.map((r) => ({ acc: r.acc, load: r.load, drop: r.dropRate })),
  };
}

type Scored = Pick<LabScore, 'acc' | 'maxLoad' | 'drop' | 'capacity' | 'cost'>;
/** Brief 1: no idle or hogging expert on any seed, and the work still gets done. */
export const passesBrief1 = (r: Pick<Scored, 'acc' | 'maxLoad'>) => r.maxLoad <= MAX_LOAD_BAR && r.acc >= ACC_BAR;
/** Brief 2: tight buffers (C ≤ 1.25) and almost nothing dropped. */
export const passesBrief2 = (r: Pick<Scored, 'acc' | 'drop' | 'capacity'>) => r.capacity <= TIGHT_C && r.drop <= DROP_BAR && r.acc >= ACC_BAR;
/** Brief 3 bar: Brief 2's quality at any capacity; the optional goal is the lowest k × C. */
export const passesBrief3 = (r: Pick<Scored, 'acc' | 'drop'>) => r.drop <= DROP_BAR && r.acc >= ACC_BAR;

export const allSettings = (): LabSetting[] =>
  LAB_K.flatMap((k) => LAB_ALPHA.flatMap((alpha) => LAB_CAPACITY.flatMap((capacity) => LAB_NOISE.map((noise) => ({ k, alpha, capacity, noise })))));

export const BRIEFS = [
  { id: 'balance', tab: '1 · No idle experts' },
  { id: 'tight', tab: '2 · Tight buffers' },
  { id: 'cheap', tab: '3 · Cheapest layer' },
] as const;
export type BriefId = (typeof BRIEFS)[number]['id'];

/** Evidence-only hints: they describe what the measurements show, never which setting to pick. */
export function labHint(brief: BriefId, r: Scored & { minLoad: number }): string {
  if (r.acc < ACC_BAR) return `Accuracy is ${(100 * r.acc).toFixed(1)}%, below the ${100 * ACC_BAR}% bar. Look at the per-seed results: is one seed dragging the average down, and what is its routing doing?`;
  if (brief === 'balance') {
    if (r.maxLoad > MAX_LOAD_BAR)
      return `On the worst seed the busiest expert handles ${(100 * r.maxLoad).toFixed(1)}% of tokens and the quietest ${(100 * r.minLoad).toFixed(1)}%. Nothing in the task loss cares how the work is shared: the router only learns what helps the prediction.`;
    return 'Every seed shares the tokens out evenly enough.';
  }
  if (r.drop > DROP_BAR) {
    const why =
      r.maxLoad > MAX_LOAD_BAR
        ? ' The busiest expert is far over its share, so its buffer overflows every batch.'
        : ' The load is fairly even on average, but each mini-batch is a random draw: some batches still send more than the buffer holds to one expert.';
    return `${(100 * r.drop).toFixed(1)}% of token slots are being dropped.${why}`;
  }
  if (brief === 'tight' && r.capacity > TIGHT_C) return `Almost nothing is dropped, but these buffers are larger than the brief allows (C ≤ ${TIGHT_C}).`;
  if (brief === 'cheap') return `This layer reserves ${r.cost.toFixed(2)} expert slots per token. Each slot is a full expert evaluation the hardware pays for, used or not.`;
  return 'Tight buffers, almost nothing dropped.';
}
