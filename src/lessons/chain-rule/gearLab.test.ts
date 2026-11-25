// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import {
  aim,
  initial,
  permutations,
  pipeline,
  ranked,
  report,
  searchWalks,
  revived,
  saturated,
  sensitivities,
  turn,
} from './gearLab';

const stages = [0, 1, 2, 3, 4];
function multipleRare(passes: number, total: number) {
  expect(passes).toBeGreaterThan(1);
  expect(passes / total).toBeLessThan(0.5);
}
it('checks local slopes and every complete sensitivity against centred differences', () => {
  for (const knobs of [initial, saturated, turn(saturated, 2, -3.5)]) {
    const h = 1e-5;
    const { outputs, knobSlopes, inputSlopes } = pipeline(knobs);
    stages.forEach((i) => {
      const ahead = pipeline(turn(knobs, i, h)).outputs;
      const behind = pipeline(turn(knobs, i, -h)).outputs;
      expect(knobSlopes[i]).toBeCloseTo((ahead[i] - behind[i]) / (2 * h), 8);
      expect(sensitivities(knobs)[i]).toBeCloseTo((ahead[4] - behind[4]) / (2 * h), 8);
      if (i === 0) {
        expect(inputSlopes[0]).toBeCloseTo(
          (pipeline(knobs, 0.7 + h).outputs[0] - pipeline(knobs, 0.7 - h).outputs[0]) / (2 * h),
          9,
        );
      } else {
        const prevSlope = pipeline(knobs).knobSlopes[i - 1];
        if (Math.abs(prevSlope) > 1e-6) {
          const plus = pipeline(turn(knobs, i - 1, h)).outputs;
          const minus = pipeline(turn(knobs, i - 1, -h)).outputs;
          expect(inputSlopes[i]).toBeCloseTo((plus[i] - minus[i]) / (plus[i - 1] - minus[i - 1]), 5);
        }
      }
      expect(Number.isFinite(outputs[i])).toBe(true);
    });
  }
});
it('scores ranking, one-shot control and saturation recovery across the UI grids', () => {
  const orders = [...permutations(stages)];
  multipleRare(orders.filter((order) => ranked(order, initial)).length, orders.length);
  const choices = stages.flatMap((i) =>
    Array.from({ length: 41 }, (_, j) => pipeline(turn(initial, i, (j - 20) / 100)).outputs[4]),
  );
  multipleRare(choices.filter((value) => Math.abs(value - aim) <= 0.01).length, choices.length);
  const shifts = Array.from({ length: 81 }, (_, i) => Math.abs(sensitivities(turn(saturated, 2, i / 20 - 4))[0]));
  multipleRare(shifts.filter((slope) => slope >= revived).length, shifts.length);
  expect(shifts.at(-1)).toBeLessThan(revived);
});
it('pins the counts for a backward walk and independent calculations', () => {
  let cache: (number | null)[] = [],
    products = 0,
    independent = 0;
  for (const stage of [...stages].reverse()) {
    const hit = report(initial, stage, cache, true);
    expect(hit.value).toBeCloseTo(sensitivities(initial)[stage], 12);
    cache = hit.costs;
    products += hit.work;
    independent += report(initial, stage, [], false).work;
  }
  expect(products).toBe(7);
  expect(independent).toBe(10);
  expect(searchWalks()).toEqual({ minimum: 7, solutions: 480, tried: 3840 });
});
