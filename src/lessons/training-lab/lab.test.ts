import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, Trainer, generate, paramCount, sizesOf, split, trainUntil, type Config } from './lab';
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

describe('V2-WP3: the compute-engine path reproduces Trainer.step', () => {
  it('same batches, same weights: CpuMlpEngine + adopt matches 200 CPU steps', async () => {
    const { CpuMlpEngine } = await import('../../lib/compute/cpuBackend');
    const pts = generate('spiral', 120, 0.05, 3);
    const { train, test } = split(pts, 0.5, 3);
    const cfg: Config = { ...DEFAULT_CONFIG, hidden: [6, 6], optimizer: 'adam', lr: 0.03, l2: 0.001, seed: 4 };
    const ref = new Trainer(cfg, train, test);
    const viaEngine = new Trainer(cfg, train, test);
    const eng = new CpuMlpEngine({ sizes: sizesOf(cfg), acts: ['tanh', 'tanh', 'sigmoid'], loss: 'bce', opt: { kind: 'adam' }, l2: cfg.l2 });
    eng.setWeights(0, { W: viaEngine.net.W.map((L) => Float32Array.from(L.flat())), b: viaEngine.net.b.map((b) => Float32Array.from(b)) });
    const d = viaEngine.trainingData();
    eng.setData(d.X, d.Y, d.n);
    for (let chunk = 0; chunk < 4; chunk++) {
      ref.step(50);
      const idx = viaEngine.drawBatches(50);
      eng.trainSync({ steps: 50, batch: cfg.batch, idx, lr: new Float32Array(50).fill(cfg.lr) });
      const w = eng.readWeightsSync()[0];
      viaEngine.adopt(w.W, w.b, 50);
    }
    expect(viaEngine.steps).toBe(200);
    const a = ref.metrics();
    const b = viaEngine.metrics();
    expect(b.trainLoss).toBeCloseTo(a.trainLoss, 3);
    expect(b.testAcc).toBeCloseTo(a.testAcc, 2);
    expect(viaEngine.net.W[0][0][0]).toBeCloseTo(ref.net.W[0][0][0], 3);
  });
});
