// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for vectors: three ways to hand the ten
 * emails from the scatter plot to a computer, each tried on that same data.
 */
import type { Rival } from '../../components/RivalTable';
import { EMAILS } from './EmailSpace';

/** Rival (a): a hand-written rule on one measurement. Errors for "links > t". */
export function ruleErrors(t: number, emails = EMAILS): number {
  return emails.filter((e) => e.links > t !== e.spam).length;
}

/** The best any single-measurement threshold rule can do (links or exclamation marks, every threshold). Ties keep the first found. */
export function bestSingleRule(emails = EMAILS): { key: 'links' | 'bangs'; t: number; errors: number; wrong: string[] } {
  let best = { key: 'links' as 'links' | 'bangs', t: 0, errors: Infinity, wrong: [] as string[] };
  for (const key of ['links', 'bangs'] as const)
    for (let t = -1; t <= 10; t++) {
      const wrong = emails.filter((e) => e[key] > t !== e.spam).map((e) => e.text);
      if (wrong.length < best.errors) best = { key, t, errors: wrong.length, wrong };
    }
  return best;
}

/** Rival (b): squash both measurements into one number (their total). Pairs of emails that become identical. */
export function squashCollisions(emails = EMAILS): { pairs: number; total: number } {
  let pairs = 0;
  let total = 0;
  for (let i = 0; i < emails.length; i++)
    for (let j = i + 1; j < emails.length; j++) {
      total++;
      if (emails[i].links + emails[i].bangs === emails[j].links + emails[j].bangs) pairs++;
    }
  return { pairs, total };
}

/** Does the total separate spam from real mail on this data? (Lowest spam total minus highest real total.) */
export function squashGap(emails = EMAILS): number {
  const s = (e: (typeof EMAILS)[number]) => e.links + e.bangs;
  return Math.min(...emails.filter((e) => e.spam).map(s)) - Math.max(...emails.filter((e) => !e.spam).map(s));
}

/** Rival (c): keep the vector. How many emails have a nearest neighbour (straight-line distance) with their own label? */
export function nearestAgree(emails = EMAILS): number {
  return emails.filter((e, i) => {
    let best = Infinity;
    let label = e.spam;
    emails.forEach((o, j) => {
      if (j === i) return;
      const d = Math.hypot(e.links - o.links, e.bangs - o.bangs);
      if (d < best - 1e-12) {
        best = d;
        label = o.spam;
      }
    });
    return label === e.spam;
  }).length;
}

const FLIGHT = EMAILS.findIndex((e) => e.text.startsWith('Flight'));
const BIRTHDAY = EMAILS.findIndex((e) => e.text.startsWith('Happy birthday'));
export const pairDistance = () => Math.hypot(EMAILS[FLIGHT].links - EMAILS[BIRTHDAY].links, EMAILS[FLIGHT].bangs - EMAILS[BIRTHDAY].bangs);

export function rivals(): Rival[] {
  const sq = squashCollisions();
  return [
    {
      name: 'A hand-written rule',
      result: `“More than 3 links means spam” gets ${ruleErrors(3)} of the ${EMAILS.length} emails wrong: it misses “YOU HAVE WON!!!” and “Earn money from home!!!!”. The best threshold on any single measurement, “more than ${bestSingleRule().t} exclamation mark”, still gets ${bestSingleRule().errors} wrong (the birthday card). And the rule’s one knob moves in whole steps: nudge it and nothing changes until a decision suddenly flips. Someone has to rewrite it for every new trick.`,
      verdicts: ['partly', 'fail', 'pass'],
    },
    {
      name: 'Squash it into one number',
      result: `Add links and exclamation marks into one “suspiciousness” total. On these ten emails it happens to separate spam from real mail, but ${sq.pairs} of the ${sq.total} pairs of emails become identical. The flight confirmation [3, 0] and the birthday card [0, 3] both score 3, though they are ${pairDistance().toFixed(2)} apart as vectors. Whatever later needs to treat links and exclamation marks differently can’t.`,
      verdicts: ['fail', 'partly', 'pass'],
    },
    {
      name: 'A list of numbers: a vector',
      result: `Keeps every measurement, so “similar” becomes “close”: ${nearestAgree()} of the ${EMAILS.length} emails have a nearest neighbour with the same label. Each entry is a number a later weight can scale smoothly, and the arithmetic is the same handful of multiply-adds whether there are 2 entries or 4,096.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
