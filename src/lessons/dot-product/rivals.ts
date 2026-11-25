// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for the dot product: five ways to turn
 * an email's counts [links, exclamation marks, mentions of your name] into one
 * spam score, each tried on the same eleven emails: the lab's ten, plus a
 * real email that looks like spam except that it mentions you by name (not
 * the lab's bank statement, so brief 3 stays fresh). A ranking is right when
 * every spam email scores above every real one.
 */
import type { Rival } from '../../components/RivalTable';
import { MAILS, dot, type Mail } from './spamLab';

/** A real email: as many links and exclamation marks as the cruise spam [4, 6, 0], but it uses your name twice. */
export const TICKETS: Mail = { subject: 'Sam, tickets for Saturday!!!', spam: false, x: [4, 6, 2] };
export const EMAILS: Mail[] = [...MAILS, TICKETS];
/** The lesson's hand-picked weights: links +1, exclamation marks +2, your name −3. */
export const W = [1, 2, -3];

export type Score = (x: number[]) => number;

export const SCORERS: Record<'count' | 'product' | 'max' | 'distance' | 'dot', Score> = {
  /** Count the spammy features (positive weight) that are "high", i.e. at least 3. */
  count: (x) => x.filter((v, i) => W[i] > 0 && v >= 3).length,
  /** Multiply all the counts together. */
  product: (x) => x.reduce((a, b) => a * b, 1),
  /** Only the strongest single piece of weighted evidence. */
  max: (x) => Math.max(...x.map((v, i) => v * W[i])),
  /** Closeness to the average spam email (minus the distance, so higher is spammier). */
  distance: (x) => -Math.hypot(...x.map((v, i) => v - typicalSpam()[i])),
  /** The weighted sum. */
  dot: (x) => dot(W, x),
};

export function typicalSpam(): number[] {
  const spam = EMAILS.filter((m) => m.spam);
  return [0, 1, 2].map((i) => spam.reduce((a, m) => a + m.x[i], 0) / spam.length);
}

/** The worst offence of a scorer: the real email with the highest score, against the spam email with the lowest. */
export function worstPair(score: Score): { spam: Mail; real: Mail; spamScore: number; realScore: number; correct: boolean } {
  const spam = EMAILS.filter((m) => m.spam).reduce((a, b) => (score(b.x) < score(a.x) ? b : a));
  const real = EMAILS.filter((m) => !m.spam).reduce((a, b) => (score(b.x) > score(a.x) ? b : a));
  return { spam, real, spamScore: score(spam.x), realScore: score(real.x), correct: score(spam.x) > score(real.x) };
}

const r2 = (v: number) => (Math.round(v * 100) / 100).toString();

export function rivals(): Rival[] {
  const c = SCORERS;
  const parcel = EMAILS.find((m) => m.subject.startsWith('Your parcel'))!;
  const news = EMAILS.find((m) => m.subject === 'Weekly newsletter')!;
  const ts = typicalSpam();
  const d = worstPair(c.dot);
  return [
    {
      name: 'Count the strong signals',
      result: `Count the spammy features that are “high” (3 or more). The tickets email [4, 6, 2] scores ${c.count(TICKETS.x)}, more than the parcel scam [7, 2, 0] with ${c.count(parcel.x)}, which ties with the weekly newsletter [4, 1, 0] at ${c.count(news.x)}: 7 links counts no more than 4, and your name counts for nothing. Counts jump in whole steps, so there is no slope to follow.`,
      verdicts: ['fail', 'fail', 'pass'],
    },
    {
      name: 'Multiply the features',
      result: `None of the five spam emails mentions your name, so every one of them scores ${c.product(EMAILS.find((m) => m.spam)!.x)}: one zero wipes out all the other evidence. The tickets email scores ${c.product(TICKETS.x)}, the highest of all. The slope for each feature is the product of all the others, so it vanishes whenever any of them is 0.`,
      verdicts: ['fail', 'fail', 'pass'],
    },
    {
      name: 'Take the biggest feature',
      result: `Score by the largest single weighted count. The tickets email scores ${c.max(TICKETS.x)}, above the parcel scam’s ${c.max(parcel.x)}: your name never gets a say. Only the winning feature has a slope; every other weight gets none.`,
      verdicts: ['fail', 'partly', 'pass'],
    },
    {
      name: 'Distance to a typical spam',
      result: `Measure how close each email is to the average spam, [${ts.map(r2).join(', ')}]. The tickets email is ${r2(-c.distance(TICKETS.x))} away, closer than the parcel scam at ${r2(-c.distance(parcel.x))}. Distance knows “alike” but not “for” or “against”, and it needs that typical point first. (Moving the point far enough can fix this ranking, but then the distance starts behaving like a weighted sum.)`,
      verdicts: ['partly', 'pass', 'pass'],
    },
    {
      name: 'The weighted sum: a dot product',
      result: `With weights [1, 2, −3], the lowest-scoring spam (“${d.spam.subject}”) scores ${d.spamScore}, and the highest-scoring real email scores ${d.realScore}: every spam email ranks above every real one. The sign of each weight carries “for” or “against”, the slope for weight i is just input i, and it is nothing but multiply-adds.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
