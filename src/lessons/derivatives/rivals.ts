// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for derivatives: how many runs of the
 * model each way of tuning knobs needs, for 1, 10 and 1,000,000 knobs.
 *
 * The goal for the searches: get every knob within 0.05 of its best value,
 * with each knob ranging over 0 to 1. A random guess lands in that 0.1-wide
 * window with chance 0.1 per knob, so it takes 10^n guesses on average; a grid
 * with 10 values per knob (0.05, 0.15, …, 0.95) is guaranteed to, in 10^n runs.
 * A slope measured by nudging costs 2 runs per knob (one each side).
 */
import type { Rival } from '../../components/RivalTable';

export const KNOBS = [1, 10, 1_000_000];

/** Runs as a power of ten (exponent), for counts too large to write out. */
export const randomSearchLog10 = (n: number) => n * -Math.log10(0.1);
export const gridLog10 = (n: number) => n * Math.log10(10);
export const slopeRuns = (n: number) => 2 * n;

/** Check the random-search figure by simulation for one knob: average guesses until one lands within 0.05 of the best value. */
export function simulateOneKnob(trials: number, seed: number, best = 0.37): number {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let total = 0;
  for (let k = 0; k < trials; k++) {
    let n = 1;
    while (Math.abs(next() - best) > 0.05) n++;
    total += n;
  }
  return total / trials;
}

const fmt = (n: number) => n.toLocaleString('en-GB');
const pow = (e: number) => (e <= 6 ? fmt(10 ** e) : `10^${fmt(e)}`);
const list = (f: (n: number) => string) => KNOBS.map(f).join(', ');

export function rivals(): Rival[] {
  return [
    {
      name: 'Random search',
      result: `Guess settings and keep the best. To get every knob within 0.05 of its best value takes about ${list((n) => pow(randomSearchLog10(n)))} runs on average for 1, 10 and 1,000,000 knobs. Each guess ignores everything the last one found out.`,
      verdicts: ['pass', 'fail', 'fail'],
    },
    {
      name: 'Try every setting on a grid',
      result: `Ten values per knob is guaranteed to get that close, in exactly ${list((n) => pow(gridLog10(n)))} runs. For 10 knobs that is ten billion runs, and the million-knob grid has more points than there are atoms in the universe (about 10^80).`,
      verdicts: ['pass', 'fail', 'fail'],
    },
    {
      name: 'Measure each slope by nudging',
      result: `Nudge each knob a little each way and see how the loss responds: ${list((n) => fmt(slopeRuns(n)))} runs give every knob its direction and its sensitivity. That is the first rival that grows in step with the number of knobs instead of exploding, but two million runs of a big model, for every single step, is still far too many.`,
      verdicts: ['pass', 'pass', 'partly'],
    },
    {
      name: 'The slope, by backpropagation',
      result: `The same slopes, worked out exactly from the derivative rules in this lesson, walking backwards through the model once. Every knob’s slope costs about as much as three runs of the model, whether there are 10 knobs or a million. That trick is backpropagation, and the chain rule is its engine.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
