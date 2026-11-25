// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { klNote, worstTerm } from './kl';
import { entropy, klDivergence } from '../../lib/prob';
import { questions } from './quiz';

describe('KL explorer note', () => {
  it('says zero when q = p', () => {
    expect(klNote([0.7, 0.2, 0.1], [0.7, 0.2, 0.1])).toContain('no extra surprise');
  });
  it('names the bigger direction: a confident belief that is wrong costs more than a vague one', () => {
    const p = [0.7, 0.2, 0.1]; // the widget's starting truth
    const q = [1 / 3, 1 / 3, 1 / 3]; // and starting (flat) belief
    expect(klDivergence(p, q).toFixed(3)).toBe('0.428');
    expect(klDivergence(q, p).toFixed(3)).toBe('0.468');
    expect(klNote(p, q)).toMatch(/^D\(q‖p\) is the bigger one/);
    expect(klNote(q, p)).toMatch(/^D\(p‖q\) is the bigger one/);
    // swapped, the confident p is the belief, and its costliest term is rain (0.1 believed, 1/3 true)
    expect(worstTerm(q, p).i).toBe(2);
  });
  it('explains infinity when the belief rules out something that happens', () => {
    expect(klNote([0.5, 0.5, 0], [1, 0, 0])).toContain('infinitely surprised');
  });
});

describe('the weather example in the prose', () => {
  it('a 50/50 belief when it is sunny 90% of the time: 1 bit = 0.469 unavoidable + 0.531 extra', () => {
    expect(entropy([0.9, 0.1], 2).toFixed(3)).toBe('0.469');
    expect(klDivergence([0.9, 0.1], [0.5, 0.5]).toFixed(3)).toBe('0.531');
    expect(klDivergence([0.5, 0.5], [0.9, 0.1]).toFixed(3)).toBe('0.737');
  });
});

describe('quiz information.kl', () => {
  it('answer is 2 − H(p) for every p it can draw', () => {
    const q = questions.find((x) => x.id === 'information.kl')!;
    expect(q).toBeTruthy();
    const seen = new Set<number>();
    for (let k = 0; k < 4; k++) {
      const made = (q as unknown as { make: (r: { int: (n: number) => number }) => { answer: number; prompt: string } }).make({ int: () => k });
      seen.add(made.answer);
    }
    expect([...seen].sort()).toEqual([0, 0.25, 0.5, 1]);
    for (const p of [[1 / 2, 1 / 4, 1 / 8, 1 / 8], [1 / 2, 1 / 2, 0, 0], [1 / 2, 1 / 4, 1 / 4, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
      expect(klDivergence(p, [0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2 - entropy(p, 2), 12);
  });
});

describe('the misconception: KL is not a distance', () => {
  it('breaks the triangle inequality: going via q can cost less than going directly', () => {
    const p = [0.9, 0.1];
    const q = [0.5, 0.5];
    const r = [0.1, 0.9];
    expect(klDivergence(p, r)).toBeGreaterThan(klDivergence(p, q) + klDivergence(q, r));
  });
});
