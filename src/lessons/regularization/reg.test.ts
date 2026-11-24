import { describe, expect, it } from 'vitest';
import { NO_CURE, Run, TEST_SET, coefMse, ridgeFit, trainAndScore, trainSet, valSet } from './reg';
import { RESULTS } from './results';
import { DATASETS, GRID, runSetting } from './search';
import { winners } from './Widgets';

describe('ridge regression', () => {
  it('λ = 0 interpolates 12 points with degree 11; larger λ shrinks the coefficients', () => {
    const tr = trainSet(1);
    expect(coefMse(ridgeFit(tr, 11, 0), tr)).toBeLessThan(1e-10);
    let prev = Infinity;
    for (const l of [1e-6, 1e-4, 1e-2, 1]) {
      const c = ridgeFit(tr, 11, l);
      const size = c.slice(1).reduce((t, v) => t + v * v, 0);
      expect(size).toBeLessThan(prev);
      prev = size;
    }
  });
});

/** Every number quoted in the regularisation lesson, pinned. */
describe('numbers quoted in the regularisation lesson', () => {
  it('weight decay on the degree-11 polynomial', () => {
    const tr = trainSet(1);
    const at = (l: number) => {
      const c = ridgeFit(tr, 11, l);
      return { test: coefMse(c, TEST_SET), size: c.slice(1).reduce((t, v) => t + v * v, 0), val: coefMse(c, valSet(1)) };
    };
    expect(at(0).test).toBeCloseTo(70.7, 1);
    expect(at(0).size).toBeCloseTo(814, 0);
    const l = 10 ** -1.75; // ≈ 0.018
    expect(at(l).test).toBeCloseTo(0.158, 3);
    expect(at(l).size).toBeCloseTo(3.1, 1);
    expect(at(1).test).toBeCloseTo(0.68, 2);
    // λ ≈ 0.018 is the best on validation among the widget's quarter-decade grid
    const grid = Array.from({ length: 37 }, (_, i) => 10 ** (-8 + i * 0.25));
    const best = grid.reduce((b, g) => (at(g).val < at(b).val ? g : b));
    expect(best).toBeCloseTo(10 ** -1.75, 9);
  });

  it('early stopping on dataset 1: patience 50 stops too early; patience 200 keeps step 320', () => {
    const p50 = trainAndScore(1, NO_CURE, 50);
    expect(p50.stoppedAt).toBe(110);
    expect(p50.test).toBeCloseTo(0.189, 3);
    const p200 = trainAndScore(1, NO_CURE, 200);
    expect(p200.stoppedAt).toBe(520);
    expect(p200.test).toBeCloseTo(0.130, 3);
    expect(trainAndScore(1, NO_CURE, null).test).toBeCloseTo(0.213, 3);
  }, 30000);

  it('the cure bench defaults on dataset 1 after 4,000 steps', () => {
    const none = new Run(trainSet(1), NO_CURE, 1);
    none.step(4000);
    expect(none.errorOn(TEST_SET)).toBeCloseTo(0.213, 3);
    expect(none.weightNorm2()).toBeCloseTo(341, 0);
    const wd = new Run(trainSet(1), { l2: 0.01, dropout: 0 }, 1);
    wd.step(4000);
    expect(wd.errorOn(wd.train)).toBeGreaterThan(0.05);
    expect(wd.weightNorm2()).toBeCloseTo(5, 0);
    const weak = new Run(trainSet(1), { l2: 1e-4, dropout: 0 }, 1);
    weak.step(4000);
    expect(weak.errorOn(weak.train)).toBeLessThan(1e-3);
    const dr = new Run(trainSet(1), { l2: 0, dropout: 0.3 }, 1);
    dr.step(4000);
    expect(dr.weightNorm2()).toBeGreaterThan(300);
    expect(dr.errorOn(TEST_SET)).toBeLessThan(0.2);
  }, 30000);

  it('the recorded fair comparison: winners and the claims about them', () => {
    const ws = winners();
    const by = Object.fromEntries(ws.map((w) => [w.id, w.best]));
    expect(by.none.test.toFixed(3)).toBe('0.251');
    expect(by.early.label).toBe('patience 400');
    expect(by.early.test).toBeCloseTo(0.123, 3);
    expect(by.l2.label).toBe('λ = 0.01');
    expect(by.l2.test).toBeCloseTo(0.128, 3);
    expect(by.dropout.label).toBe('p = 0.5');
    expect(by.dropout.test).toBeCloseTo(0.135, 3);
    const t = (m: string) => by[m].perDataset.map((o) => o.test);
    const wins = (a: number[], b: number[]) => a.filter((v, i) => v < b[i]).length;
    for (const m of ['early', 'l2', 'dropout']) expect(wins(t(m), t('none'))).toBe(DATASETS);
    expect(wins(t('early'), t('l2'))).toBe(6);
    expect(wins(t('l2'), t('dropout'))).toBe(5);
    // patience 100 did slightly better on test than the validation-chosen 400
    const p100 = RESULTS.find((r) => r.label === 'patience 100')!;
    expect(p100.test).toBeLessThan(by.early.test);
    expect(p100.val).toBeGreaterThan(by.early.val);
  });

  it('the recorded search is genuine: re-training a sample of its runs reproduces them', () => {
    // one run per method family, on two datasets (the full search takes minutes; see search.run.ts)
    for (const label of ['no cure', 'patience 400', 'λ = 0.01', 'p = 0.5']) {
      const setting = GRID.find((g) => g.label === label)!;
      const row = RESULTS.find((r) => r.label === label)!;
      for (const k of [1, 7]) {
        const o = runSetting(setting, k);
        expect(o.test).toBeCloseTo(row.perDataset[k - 1].test, 5);
        expect(o.val).toBeCloseTo(row.perDataset[k - 1].val, 5);
      }
    }
  }, 60000);
});
