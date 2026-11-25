// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { BANK, LONG, MAILS, bands, cosine, gap, sweep, sweep3, unit } from './spamLab';

describe('Spam Scorer data', () => {
  it('has 5 spam and 5 real emails, and the long copy doubles the newsletter', () => {
    expect(MAILS.filter((m) => m.spam).length).toBe(5);
    expect(MAILS.length).toBe(10);
    expect(LONG.x).toEqual([8, 2, 0]);
    expect(BANK.spam).toBe(false);
  });
  it('a ranking depends only on the direction of w (scaling w scales the gap)', () => {
    expect(gap([2, 2], MAILS)).toBeCloseTo(2 * gap([1, 1], MAILS), 12);
  });
});

describe('Spam Scorer solver (angle of w swept in 0.01° steps)', () => {
  const s1 = sweep(MAILS);
  it('brief 1: one band of directions ranks perfectly, 9.47° to 81.86°', () => {
    expect(bands(s1.pass)).toEqual([[9.47, 81.86]]);
    // a range of answers, but a random direction usually fails (about 20% pass)
    expect(s1.pass.length / 36000).toBeGreaterThan(0.1);
    expect(s1.pass.length / 36000).toBeLessThan(0.3);
  });
  it('brief 2: the widest gap is 2.967 at 38.66°, and many directions reach 95% of it', () => {
    expect(s1.best.deg).toBe(38.66);
    expect(s1.best.gap.toFixed(3)).toBe('2.967');
    let within = 0;
    for (let k = 0; k < 36000; k++) if (gap(unit(k / 100), MAILS) >= 0.95 * s1.best.gap) within++;
    expect(within).toBeGreaterThan(100); // more than one answer (over a degree of directions)
  });
  it('brief 3: every perfect ranking needs a negative weight on name mentions', () => {
    expect(bands(sweep([...MAILS, BANK]).pass)).toEqual([]); // impossible on links and exclamation marks alone
    const s3 = sweep3([...MAILS, BANK]);
    expect(s3.posOrZero).toBe(0);
    expect(s3.negative).toBe(5944);
  });
  it('brief 4: no dot-product direction works with the long copy; cosine works from 40.42° to 52.97°', () => {
    expect(sweep([...MAILS, LONG], 'dot').pass.length).toBe(0);
    expect(bands(sweep([...MAILS, LONG], 'cosine').pass)).toEqual([[40.42, 52.97]]);
    // cosine ignores length: the copy scores exactly like the original
    expect(cosine([1, 1], LONG.x.slice(0, 2))).toBeCloseTo(cosine([1, 1], MAILS[8].x.slice(0, 2)), 12);
  });
});
