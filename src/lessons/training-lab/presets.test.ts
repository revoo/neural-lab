import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, Trainer, generate, split } from './lab';
import { PRESETS } from './presets';

/** The lesson text describes what each preset does; these tests pin those claims down. */
const run = (key: string, steps: number) => {
  const p = PRESETS[key];
  const { train, test } = split(generate(p.dataset, p.n, p.noise, 1), p.trainFrac, 1);
  const t = new Trainer({ ...DEFAULT_CONFIG, ...p.cfg }, train, test);
  t.step(steps);
  return t.metrics();
};

describe('lesson presets behave as the text says', () => {
  it('a single neuron on raw inputs cannot do the circle', () => expect(run('linear', 2000).testAcc).toBeLessThan(0.6));
  it('with x² and y² it can', () => expect(run('squares', 100).testAcc).toBeGreaterThanOrEqual(0.95));
  it('the raw spiral is learnable with ReLU 8×8×8 at step size 0.01', () => expect(run('spiralRaw', 3000).testAcc).toBeGreaterThan(0.95));
  it('momentum at step size 1 gets stuck near chance on the moons', () => expect(run('hot', 2000).testAcc).toBeLessThan(0.6));
  it('few noisy points and a big network: train accuracy far above test accuracy', () => {
    const m = run('overfit', 1000);
    expect(m.trainAcc - m.testAcc).toBeGreaterThan(0.3);
    expect(m.testLoss).toBeGreaterThan(3 * m.trainLoss);
  });
});
