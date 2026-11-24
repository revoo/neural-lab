import { describe, expect, it } from 'vitest';
import { cost, run, score, type Circuit } from './workbench';

const xor = (a: number, b: number) => a ^ b;

describe('logic workbench', () => {
  it('a single neuron solves AND', () => {
    const c: Circuit = { hidden: [], out: { w: [1, 1], b: -1.5 } };
    expect(score(c, (a, b) => a & b).solved).toBe(true);
  });
  it('a cost-3 XOR circuit exists with half-unit weights', () => {
    const c: Circuit = {
      hidden: [
        { w: [0.5, -0.5], b: 0 }, // A and not B
        { w: [-0.5, 0.5], b: 0 }, // B and not A
      ],
      out: { w: [0.5, 0.5], b: 0 }, // OR
    };
    expect(score(c, xor).solved).toBe(true);
    expect(cost(c)).toBe(3);
  });
  it('no single step neuron on a grid of weights solves XOR', () => {
    const vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
    for (const w1 of vals) for (const w2 of vals) for (const b of vals) {
      expect(score({ hidden: [], out: { w: [w1, w2], b } }, xor).solved).toBe(false);
    }
  });
  it('exposes hidden activations', () => {
    const c: Circuit = { hidden: [{ w: [1, 0], b: -0.5 }], out: { w: [1], b: -0.5 } };
    expect(run(c, 1, 0)).toMatchObject({ hidden: [1], out: 1 });
  });
});
