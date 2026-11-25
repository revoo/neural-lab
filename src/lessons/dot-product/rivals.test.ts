// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { EMAILS, SCORERS, TICKETS, W, rivals, typicalSpam, worstPair } from './rivals';

const find = (s: string) => EMAILS.find((m) => m.subject.startsWith(s))!;

describe('dot-product: why a weighted sum, and not something else', () => {
  it('each rival misranks the tickets email against a spam email', () => {
    const parcel = find('Your parcel');
    const news = find('Weekly newsletter');
    expect(SCORERS.count(TICKETS.x)).toBe(2);
    expect(SCORERS.count(parcel.x)).toBe(1);
    expect(SCORERS.count(news.x)).toBe(1);
    expect(EMAILS.filter((m) => m.spam).every((m) => SCORERS.product(m.x) === 0)).toBe(true);
    expect(SCORERS.product(TICKETS.x)).toBe(48);
    expect(Math.max(...EMAILS.map((m) => SCORERS.product(m.x)))).toBe(48);
    expect(SCORERS.max(TICKETS.x)).toBe(12);
    expect(SCORERS.max(parcel.x)).toBe(7);
    expect(typicalSpam()).toEqual([5, 4.8, 0]);
    expect((-SCORERS.distance(TICKETS.x)).toFixed(2)).toBe('2.54');
    expect((-SCORERS.distance(parcel.x)).toFixed(2)).toBe('3.44');
    for (const k of ['count', 'product', 'max', 'distance'] as const) {
      expect(worstPair(SCORERS[k]).correct, k).toBe(false);
      expect(worstPair(SCORERS[k]).real, k).toBe(TICKETS);
    }
    expect(EMAILS.some((m) => m.subject === 'Sam, your statement is ready!!!')).toBe(false);
  });

  it('a typical point moved far away can fix the distance ranking', () => {
    const far = [5, 5, -10];
    const d = (x: number[]) => -Math.hypot(...x.map((v, i) => v - far[i]));
    expect(worstPair(d).correct).toBe(true);
  });

  it('the dot product ranks every spam above every real email', () => {
    expect(W).toEqual([1, 2, -3]);
    const p = worstPair(SCORERS.dot);
    expect(p).toMatchObject({ spamScore: 11, realScore: 10, correct: true })
    expect(p.real).toBe(TICKETS);;
    expect(p.spam.subject).toBe('Your parcel is held, click here');
  });

  it('only the dot product passes all three tests', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['The weighted sum: a dot product']);
    expect(rs.every((r) => !/NaN|undefined/.test(r.result))).toBe(true);
  });
});
