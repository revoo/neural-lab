// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for the chain rule: three ways to get
 * the slope of the loss for every weight in a pipeline, counted in runs of
 * the pipeline, plus an accuracy check on the lesson's own neuron
 * (w → z = 1.5w + 0.2 → a = tanh z → L = (a − 0.8)², at w = 0.4).
 */
import type { Rival } from '../../components/RivalTable';
import { forward } from './ChainExplorer';

export const WEIGHTS = [5, 1000, 1_000_000];
/** Nudge every weight and re-run: one base run plus one per weight. */
export const nudgeRuns = (n: number) => n + 1;
/** The chain rule walked forwards from each weight: one walk per weight, each about as dear as a run. */
export const forwardWalks = (n: number) => n;
/** One forward run, remembering every value, then one walk backwards: about three runs in all, for any n. */
export const backwardRuns = (_n: number) => 3;

export const W0 = 0.4;
/** The chain rule's slope for the lesson's neuron: the product of the local derivatives. */
export const chainSlope = (w = W0) => {
  const s = forward(w);
  return s.dz * s.da * s.dL;
};
/** The slope measured by nudging w by h and re-running (one-sided, as in "nudge and re-run"). */
export const nudgedSlope = (h: number, w = W0) => (forward(w + h).L - forward(w).L) / h;

const fmt = (n: number) => n.toLocaleString('en-GB');
const f4 = (v: number) => v.toFixed(4).replace('-', '−');
const list = (f: (n: number) => number) => WEIGHTS.map((n) => fmt(f(n))).join(', ');

export function rivals(): Rival[] {
  const exact = chainSlope();
  const rough = nudgedSlope(0.01);
  return [
    {
      name: 'Nudge every weight and re-run',
      result: `For 5, 1,000 and 1,000,000 weights that is ${list(nudgeRuns)} runs of the whole pipeline, and each slope is only an estimate: on this lesson’s neuron at w = ${W0}, a nudge of 0.01 measures ${f4(rough)} where the true slope is ${f4(exact)}.`,
      verdicts: ['pass', 'partly', 'fail'],
    },
    {
      name: 'The chain rule, walked forwards',
      result: `Start at one weight and carry its gear ratio forwards through every stage to the loss. Exact, but it answers for one weight per walk: ${list(forwardWalks)} walks. (This “forward mode” wins when there are few inputs and many outputs, the opposite of training.)`,
      verdicts: ['pass', 'pass', 'fail'],
    },
    {
      name: 'The chain rule, walked backwards',
      result: `Run forwards once, remembering every value, then walk back from the loss multiplying gear ratios. Every stage passes its blame to everything before it, so one walk serves every weight: about ${backwardRuns(0)} runs’ worth of work, whether there are 5 weights or a million. Exact, too: ${f4(exact)}.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
