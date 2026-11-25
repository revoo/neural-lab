// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import { accepts, experiments, secant, survey } from './sensitivityLab';

it('estimates the smooth mathematical slopes and pins the successful float32 central steps', () => {
  expect(survey(0)[1].methods[1].value).toBeCloseTo(1.53505, 10);
  expect(survey(1)[1].methods[1].passes).toBe(true);
  expect(survey(2)[0].methods[2].value).toBeCloseTo(-0.6, 12);
  expect(survey(2)[0].methods[0].value).toBeCloseTo(1.7, 12);
  expect(survey(3)[6].methods[1].passes).toBe(false);
  expect(
    survey(3)
      .filter((row) => row.methods[1].passes)
      .map((row) => row.h),
  ).toEqual([0.01, 0.001]);
});

it('has multiple probe strategies, but arbitrary probes or a wrong-side corner report generally fail', () => {
  for (const index of [0, 1, 3]) {
    const { at, run } = experiments[index];
    const slope = (x: number, y: number) => secant({ x, y: run(x) }, { x: y, y: run(y) });
    const guesses = Array.from({ length: 41 }, (_, i) => at + (i - 20) * 0.1);
    const arbitrary = guesses.flatMap((x) => guesses.map((y) => slope(x, y)));
    expect(arbitrary.filter((value) => accepts(index, [value])).length / arbitrary.length).toBeLessThan(0.5);
    expect(survey(index).filter((row) => row.methods[1].passes).length).toBeGreaterThan(1);
  }
  const choices = [-0.6, 0, 0.55, 1.7];
  expect(choices.flatMap((a) => choices.map((b) => accepts(2, [a, b]))).filter(Boolean)).toHaveLength(1);
  expect(survey(2).filter((row) => row.methods[0].passes && row.methods[2].passes).length).toBeGreaterThan(1);
});
