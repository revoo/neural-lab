import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, Trainer, generate, paramCount, split, trainUntil } from './lab';
import { BEST_FOUND, CHALLENGES, challengeData } from './challenges';
import { searchSpace } from './search';

describe('datasets', () => {
  it('are balanced, deterministic and inside the domain', () => {
    for (const name of ['circle', 'xor', 'clusters', 'spiral', 'moons'] as const) {
      const a = generate(name, 200, 0.1, 1);
      expect(a.filter((p) => p.label === 1).length).toBe(100);
      expect(generate(name, 200, 0.1, 1)).toEqual(a);
      expect(a.every((p) => Math.abs(p.x) <= 1.6 && Math.abs(p.y) <= 1.6)).toBe(true);
    }
  });
  it('split keeps every point exactly once', () => {
    const pts = generate('moons', 100, 0.1, 2);
    const { train, test } = split(pts, 0.7, 2);
    expect(train.length).toBe(70);
    expect(test.length).toBe(30);
    expect(new Set([...train, ...test]).size).toBe(100);
  });
});

describe('trainer', () => {
  it('is reproducible', () => {
    const { train, test } = split(generate('circle', 120, 0.1, 1), 0.5, 1);
    const a = new Trainer({ ...DEFAULT_CONFIG }, train, test);
    const b = new Trainer({ ...DEFAULT_CONFIG }, train, test);
    a.step(200);
    b.step(200);
    expect(a.metrics()).toEqual(b.metrics());
  });
  it('learns the circle with a hidden layer, but not with a single neuron on raw inputs', () => {
    const { train, test } = split(generate('circle', 200, 0.1, 1), 0.5, 1);
    const deep = new Trainer({ ...DEFAULT_CONFIG }, train, test);
    deep.step(1500);
    expect(deep.metrics().testAcc).toBeGreaterThan(0.95);
    const flat = new Trainer({ ...DEFAULT_CONFIG, hidden: [] }, train, test);
    flat.step(1500);
    expect(flat.metrics().testAcc).toBeLessThan(0.75);
  });
  it('counts parameters', () => {
    expect(paramCount({ ...DEFAULT_CONFIG, features: ['x', 'y'], hidden: [4, 4] })).toBe(2 * 4 + 4 + 4 * 4 + 4 + 4 + 1);
    expect(paramCount({ ...DEFAULT_CONFIG, features: ['x2', 'y2'], hidden: [] })).toBe(3);
  });
});

describe('challenge references', () => {
  for (const c of CHALLENGES) {
    it(`${c.id}: the recorded best configuration really reaches the target`, () => {
      const best = BEST_FOUND[c.id];
      expect(c.rules(best.cfg)).toBeNull();
      expect(paramCount(best.cfg)).toBe(best.params);
      const { train, test } = challengeData(c);
      const r = trainUntil(best.cfg, train, test, c.target);
      expect(r.reachedAt).toBe(best.reachedAt);
      // it is part of the documented search space
      expect(searchSpace(c).some((cfg) => JSON.stringify({ ...cfg, seed: best.cfg.seed }) === JSON.stringify(best.cfg))).toBe(true);
    });
  }
});
