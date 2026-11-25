// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for softmax: five ways to turn the
 * explorer's starting scores (cat 2.0, dog 1.0, fox 0.1, owl −1.0) into
 * probabilities, each tried on those same scores.
 */
import type { Rival } from '../../components/RivalTable';
import { softmax } from '../../lib/prob';

export const CLASSES = ['cat', 'dog', 'fox', 'owl'];
export const Z = [2.0, 1.0, 0.1, -1.0];

export const RULES = {
  divide: (z: number[]) => {
    const s = z.reduce((a, b) => a + b, 0);
    return z.map((v) => v / s);
  },
  clamp: (z: number[]) => {
    const c = z.map((v) => Math.max(0, v));
    const s = c.reduce((a, b) => a + b, 0);
    return c.map((v) => v / s);
  },
  argmax: (z: number[]) => {
    const i = z.indexOf(Math.max(...z));
    return z.map((_, j) => (j === i ? 1 : 0));
  },
  sigmoid: (z: number[]) => z.map((v) => 1 / (1 + Math.exp(-v))),
  softmax: (z: number[]) => softmax(z),
};
export type RuleId = keyof typeof RULES;

/** Slope of output `i` with respect to score `j`, by central differences. */
export function slope(rule: RuleId, i: number, j: number, z = Z, h = 1e-5): number {
  const up = z.slice();
  const dn = z.slice();
  up[j] += h;
  dn[j] -= h;
  return (RULES[rule](up)[i] - RULES[rule](dn)[i]) / (2 * h);
}
export const total = (p: number[]) => p.reduce((a, b) => a + b, 0);
/** Largest change in any output when every score is raised by `c`. */
export const shiftChange = (rule: RuleId, c: number, z = Z) => {
  const p = RULES[rule](z);
  const q = RULES[rule](z.map((v) => v + c));
  return Math.max(...p.map((v, i) => Math.abs(v - q[i])));
};

const p2 = (v: number) => (Math.round(v * 100) / 100).toFixed(2).replace('-', '−');
const show = (p: number[]) => `[${p.map(p2).join(', ')}]`;

export function rivals(): Rival[] {
  const R = RULES;
  return [
    {
      name: 'Divide by the sum',
      result: `${show(R.divide(Z))}: owl gets a “probability” of ${p2(R.divide(Z)[3])}. And if the scores ever sum to zero, the rule divides by zero.`,
      verdicts: ['fail', 'fail', 'pass'],
    },
    {
      name: 'Clamp negatives to zero, then divide',
      result: `${show(R.clamp(Z))}: owl is forced to exactly 0, and its slope is dead. Nudging owl’s score changes its probability by ${Math.abs(slope('clamp', 3, 3)).toFixed(0)}, so training can never rescue it.`,
      verdicts: ['fail', 'fail', 'pass'],
    },
    {
      name: 'Pick the top score (argmax)',
      result: `${show(R.argmax(Z))}: certain about cat, with no room for doubt. Nudge any score slightly and nothing moves: every slope is ${Math.abs(slope('argmax', 0, 0)).toFixed(0)}.`,
      verdicts: ['fail', 'fail', 'pass'],
    },
    {
      name: 'A sigmoid for each class',
      result: `${show(R.sigmoid(Z))}, which adds up to ${total(R.sigmoid(Z)).toFixed(2)}. Smooth and positive, but these are four separate yes/no answers, not one choice among four. (That is exactly right when an image can be both “cat” and “indoors”.)`,
      verdicts: ['fail', 'pass', 'pass'],
    },
    {
      name: 'Softmax',
      result: `${show(R.softmax(Z))}: every class positive, total exactly 1, highest score biggest. Every slope is alive (cat’s probability moves by ${slope('softmax', 0, 0).toFixed(2)} per unit of cat’s score), and adding 100 to every score changes nothing.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
