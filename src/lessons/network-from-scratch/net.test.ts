import { describe, expect, it } from 'vitest';
import { OPTIONS, REFERENCE, SLOTS, X_XOR, Y_XOR, accuracy, backward, forward, gradCheck, init, matmul, ew, train, type Choice } from './net';
import { GOLF_PAR, HIDDEN, LRS, golfPar } from './Widgets';

const all = (): Choice[] =>
  Array.from({ length: 4 ** 5 }, (_, m) => Object.fromEntries(SLOTS.map((s, i) => [s, Math.floor(m / 4 ** i) % 4])) as Choice);

describe('matrix helpers behave like NumPy', () => {
  it('matmul shape errors and broadcasting', () => {
    expect(() => matmul([[1, 2]], [[1, 2]])).toThrow(/not aligned/);
    expect(ew([[1], [2]], [[10, 20, 30]], (a, b) => a * b)).toEqual([
      [10, 20, 30],
      [20, 40, 60],
    ]);
    expect(() => ew([[1, 2]], [[1, 2, 3]], (a, b) => a + b)).toThrow(/broadcast/);
  });
});

describe('the reference backward pass', () => {
  it('passes a numerical gradient check', () => {
    const g = gradCheck(init(3, 1), REFERENCE);
    expect(g.ok && g.pass).toBe(true);
  });
  it('trains XOR: loss 0.667 → 0.0096 in 300 steps, all four right', () => {
    expect(forward(init(3, 1), X_XOR, Y_XOR).loss!).toBeCloseTo(0.667, 3);
    const r = train(REFERENCE, 3, 1, 300);
    expect(r.losses[299]).toBeCloseTo(0.0096, 4);
    expect(accuracy(r.P)).toBe(1);
  });
});

describe('the lab: numbers quoted in the debrief', () => {
  it('24 of the 1,024 combinations pass the gradient check; equivalent forms are accepted', () => {
    const P = init(3, 1);
    const correct = all().filter((c) => {
      const g = gradCheck(P, c);
      return g.ok && g.pass;
    });
    expect(correct.length).toBe(24);
    const valid = (s: (typeof SLOTS)[number]) => [...new Set(correct.map((c) => OPTIONS[s][c[s]].code))].sort();
    expect(valid('dW2')).toEqual(['(dz2.T @ h).T', 'h.T @ dz2']);
    expect(valid('dh')).toEqual(['(W2 @ dz2.T).T', 'dz2 * W2.T', 'dz2 @ W2.T']);
    expect(valid('dz1')).toEqual(['dh * (1 - h**2)', 'dh * (1 - tanh(z1)**2)']);
    // the answer is the same whichever random start is used to check it
    for (const c of correct) {
      const g = gradCheck(init(4, 7), c);
      expect(g.ok && g.pass).toBe(true);
    }
  }, 60000);
  it('two wrong backward passes still solve XOR', () => {
    const noN = train({ ...REFERENCE, dz2: 3 }, 3, 1, 300);
    expect(accuracy(noN.P)).toBe(1);
    const mseSlope = train({ ...REFERENCE, dz2: 1 }, 3, 1, 300);
    expect(accuracy(mseSlope.P)).toBe(1);
    expect(mseSlope.losses[299]).toBeCloseTo(0.10, 2);
    for (const ch of [{ ...REFERENCE, dz2: 3 }, { ...REFERENCE, dz2: 1 }]) {
      const g = gradCheck(init(3, 1), ch);
      expect(g.ok && !g.pass).toBe(true);
    }
  });
  it('shape errors are reported on the line that causes them', () => {
    const r = backward(init(3, 1), { ...REFERENCE, dz1: 3 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.slot).toBe('dz1');
  });
  it('the recorded par for the speed goal is the exhaustive best over the menus', () => {
    expect(HIDDEN.length * LRS.length).toBe(48);
    expect(golfPar()).toEqual(GOLF_PAR);
  }, 60000);
});
