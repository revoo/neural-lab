import { describe, expect, it } from 'vitest';
import { LANDSCAPES, golfPar, runOptimizer, stepsToTarget, tuneLR } from './optim';

const L = Object.fromEntries(LANDSCAPES.map((l) => [l.id, l]));

describe('landscapes', () => {
  it('gradients match finite differences', () => {
    for (const l of LANDSCAPES)
      for (const [x, y] of [[0.3, -0.7], [-1.2, 0.4], [1.5, 1.1]]) {
        const h = 1e-6;
        const [gx, gy] = l.grad(x, y);
        expect(gx).toBeCloseTo((l.f(x + h, y) - l.f(x - h, y)) / (2 * h), 5);
        expect(gy).toBeCloseTo((l.f(x, y + h) - l.f(x, y - h)) / (2 * h), 5);
      }
  });
  it('minima are where they claim, with value 0', () => {
    for (const l of LANDSCAPES) for (const [x, y] of l.minimum) expect(l.f(x, y)).toBeCloseTo(0, 12);
  });
  it('every start is far from the target', () => {
    for (const l of LANDSCAPES) expect(l.f(...l.start)).toBeGreaterThan(10 * l.target);
  });
});

describe('races', () => {
  it('momentum beats plain gradient descent in the narrow valley, each at its best step size', () => {
    expect(tuneLR(L.valley, 'momentum').steps).toBeLessThan(tuneLR(L.valley, 'sgd').steps);
  });
  it('adaptive methods cross the plateau far faster than plain gradient descent', () => {
    const sgd = tuneLR(L.plateau, 'sgd').steps;
    expect(tuneLR(L.plateau, 'adam').steps * 3).toBeLessThan(sgd);
    expect(tuneLR(L.plateau, 'rmsprop').steps * 3).toBeLessThan(sgd);
  });
  it('runOptimizer and stepsToTarget agree', () => {
    const cfg = { name: 'adam' as const, lr: 0.5, beta1: 0.9, beta2: 0.999 };
    expect(runOptimizer(L.banana, cfg, 400).reachedAt).toBe(stepsToTarget(L.banana, cfg, 400));
  });
  it('golf par is no worse than any tuned optimizer', () => {
    const par = golfPar(L.banana).steps;
    for (const n of ['sgd', 'momentum', 'rmsprop', 'adam'] as const) expect(par).toBeLessThanOrEqual(tuneLR(L.banana, n).steps);
    expect(stepsToTarget(L.banana, golfPar(L.banana).cfg)).toBe(par);
  });
});
