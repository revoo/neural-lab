import { describe, expect, it } from 'vitest';
import { nextRecord } from './progress';
import { parseHash } from './router';

describe('spaced repetition', () => {
  it('promotes on correct answers and demotes to box 1 on mistakes', () => {
    let r = nextRecord(undefined, true, 0);
    expect(r.box).toBe(2);
    r = nextRecord(r, true, 0);
    expect(r.box).toBe(3);
    expect(r.due).toBe(3 * 86400000);
    r = nextRecord(r, false, 10);
    expect(r.box).toBe(1);
    expect(r.due).toBe(10);
    expect(r.attempts).toBe(3);
    expect(r.correct).toBe(2);
  });
  it('caps at box 5', () => {
    let r = nextRecord(undefined, true, 0);
    for (let i = 0; i < 10; i++) r = nextRecord(r, true, 0);
    expect(r.box).toBe(5);
  });
});

describe('router', () => {
  it('parses hashes', () => {
    expect(parseHash('')).toEqual({ page: 'home' });
    expect(parseHash('#/learn/vectors')).toEqual({ page: 'lesson', id: 'vectors', anchor: undefined });
    expect(parseHash('#/glossary/gradient')).toEqual({ page: 'glossary', term: 'gradient' });
    expect(parseHash('#/nope')).toEqual({ page: 'notfound' });
  });
});
