// authored-by: Gemini Antigravity
/**
 * Lab: Tune the draft — brief definitions, exhaustive 48-cell summaries, and evidence-only hints.
 */
import type { DrafterId } from './spec';
import { SPEC_GRID } from './specResults';

export interface SpecBrief {
  id: 'greedy' | 'sampling' | 'cpu-impossible';
  tab: string;
  title: string;
  temperature: 0 | 1.0;
  targetSpeedup: number;
  impossible?: boolean;
  description: string;
  optGoal: string;
}

export const SPEC_BRIEFS: readonly SpecBrief[] = [
  {
    id: 'greedy',
    tab: '1 · Greedy (T = 0)',
    title: 'Brief 1 — Exact greedy speedup',
    temperature: 0,
    targetSpeedup: 1.95,
    description:
      'At temperature 0 (greedy), reach a production-scale modelled speed-up of at least 1.95× on the RTX 4090 roofline while keeping every generated character identical to plain target decoding.',
    optGoal: 'Find the highest modelled speed-up across all 16 greedy settings.',
  },
  {
    id: 'sampling',
    tab: '2 · Sampling (T = 1)',
    title: 'Brief 2 — Stochastic sampling',
    temperature: 1.0,
    targetSpeedup: 2.0,
    description:
      'At temperature 1.0, reach a production-scale modelled speed-up of at least 2.00× across the 12 held-out Grimm prompts. Watch how draft cost k·c competes with acceptance rate.',
    optGoal: 'Find the highest modelled speed-up across all 16 sampling settings.',
  },
  {
    id: 'cpu-impossible',
    tab: '3 · Your CPU clock',
    title: 'Brief 3 — Beat plain decoding on your own CPU',
    temperature: 0,
    targetSpeedup: 1.0,
    impossible: true,
    description:
      'Measure real wall-clock speed on the 29.7k-parameter tiny GPT running right here on your CPU. Find a (drafter, k) setting that runs faster than plain target decoding (> 1.00× real wall time) — or prove from the roofline why none can, and declare it impossible.',
    optGoal: 'Recognise the hardware bottleneck directly from the timing readout.',
  },
];

export function specCellFor(drafter: DrafterId, temperature: number, k: number) {
  return SPEC_GRID.find((c) => c.drafter === drafter && c.temperature === temperature && c.k === k)!;
}

export function specBriefSummary(brief: SpecBrief) {
  const cells = SPEC_GRID.filter((c) => c.temperature === brief.temperature);
  if (brief.impossible) {
    const bestCpu = [...SPEC_GRID].sort((a, b) => b.cpuWallSpeedup - a.cpuWallSpeedup)[0];
    return {
      total: SPEC_GRID.length,
      passingCount: 0,
      bestSpeedup: bestCpu.cpuWallSpeedup,
      bestCells: [bestCpu],
    };
  }
  const passing = cells.filter((c) => c.modelledSpeedup >= brief.targetSpeedup - 1e-9);
  const bestVal = Math.max(...cells.map((c) => c.modelledSpeedup));
  const bestCells = cells.filter((c) => Math.abs(c.modelledSpeedup - bestVal) < 0.005);
  return {
    total: cells.length,
    passingCount: passing.length,
    bestSpeedup: bestVal,
    bestCells,
  };
}

/** Evidence-only hint for Tune the draft: points only at observed metrics, never names a drafter or k value. */
export function specHint(brief: SpecBrief, drafter: DrafterId, k: number): string {
  const cell = specCellFor(drafter, brief.temperature, k);
  if (brief.impossible) {
    return `Look at the two numbers side by side: modelled GPU speed-up is ${cell.modelledSpeedup.toFixed(2)}×, while real CPU wall-clock ratio is ${cell.cpuWallSpeedup.toFixed(2)}×. Ask yourself whether a 116 KiB model sitting inside CPU cache is memory-bandwidth bound or compute-bound.`;
  }
  const draftTax = k * cell.costRatio;
  if (draftTax > 0.55) {
    return `Notice the denominator of the speed-up formula: running the draft that many steps per round adds ${(draftTax * 100).toFixed(0)}% of a target pass in overhead, eating the extra tokens per pass (E = ${cell.tokensPerPass.toFixed(2)}).`;
  }
  if (cell.tokensPerPass < brief.targetSpeedup) {
    return `At this lookahead the round emits only ${cell.tokensPerPass.toFixed(2)} tokens per target pass on average (acceptance ${(cell.alpha * 100).toFixed(1)}%), which caps speed-up below ${brief.targetSpeedup.toFixed(2)}× even before counting draft overhead.`;
  }
  return `Look at the trade between acceptance rate (${(cell.alpha * 100).toFixed(1)}%) and per-step draft cost (${(cell.costRatio * 100).toFixed(1)}% of a target pass): a cheaper draft keeps the denominator near 1 even when acceptance is lower.`;
}
