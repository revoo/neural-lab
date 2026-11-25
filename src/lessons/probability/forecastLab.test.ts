// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import {
  correctTemperature,
  days,
  dryDays,
  exact,
  fixedLogits,
  gap,
  normalise,
  ratio,
  roundLogits,
  scores,
  tolerance,
} from './forecastLab';

it('recovers the target, including after locking the first logit at zero', () => {
  expect(normalise(days)).toEqual([0.5, 0.3, 0.15, 0.05]);
  const a = exact(days);
  for (const reference of [0, a[0], -2]) {
    const config = exact(days, reference);
    expect(gap(scores(config, 1), days)).toBeLessThan(1e-12);
    expect(gap(scores(roundLogits(config), 1), days)).toBeLessThan(tolerance(0));
    expect(ratio(config[0], config[1], 1)).toBeCloseTo(5 / 3, 12);
  }
  expect(exact(days, a[0])[0]).toBe(0);
});
it('shows multiple solutions for all briefs, with poor odds on coarse random logits or a random temperature', () => {
  const target = exact(days);
  for (const reference of [0, target[0]]) {
    const good = roundLogits(exact(days, reference));
    expect(
      [0, 0.05, -0.05].filter(
        (shift) =>
          gap(
            scores(
              good.map((v, i) => (i === 1 ? v + shift : v)),
              1,
            ),
            days,
          ) < tolerance(0),
      ),
    ).toHaveLength(3);
  }
  const trials = Array.from({ length: 57 }, (_, i) => (i + 4) / 20);
  const matches = trials.filter((t) => gap(scores(fixedLogits, t), days) < tolerance(2));
  expect(matches).toContain(correctTemperature);
  expect(matches.length).toBeGreaterThan(1);
  expect(matches.length / trials.length).toBeLessThan(0.5);
  for (const logit of [-7, -8]) {
    const probabilities = scores([0, ...roundLogits(exact(dryDays, Math.log(0.55))).slice(1, 3), logit], 1);
    expect(probabilities[3]).toBeGreaterThan(0);
    expect(gap(probabilities, dryDays)).toBeLessThan(tolerance(3));
  }
  const grid = [-8, -6, -4, -2, 0, 2, 4];
  for (const brief of [0, 1, 3]) {
    let tried = 0,
      passed = 0;
    for (const a of brief === 1 ? [0] : grid)
      for (const b of grid)
        for (const c of grid)
          for (const d of grid) {
            tried++;
            if (gap(scores([a, b, c, d], 1), brief === 3 ? dryDays : days) < tolerance(brief)) passed++;
          }
    expect(passed / tried).toBeLessThan(0.5);
  }
});
