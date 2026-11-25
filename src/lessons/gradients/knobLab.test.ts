// authored-by: Codex Meridian 6
import { expect, it } from 'vitest';
import {
  axes,
  contourError,
  fixedStep,
  gradient,
  keyboardTrial,
  loss,
  minTurns,
  move,
  start,
  sweep,
  type Point,
} from './knobLab';

it('matches central finite differences on every axis', () => {
  for (const p of [start, [0.4, 0.7, 1.3], [-2.1, -0.8, 0.6]] as Point[]) {
    axes
      .filter((_, i) => i % 2 === 0)
      .forEach((axis, i) => {
        const h = 1e-5;
        expect(gradient(p)[i]).toBeCloseTo((loss(move(p, axis, h)) - loss(move(p, axis, -h))) / (2 * h), 9);
      });
  }
});
it('pins the fewest grid turns and the searched fixed step budget', () => {
  expect(minTurns()).toBe(17);
  const rates = sweep();
  expect(Math.min(...rates.map((row) => row.turns))).toBe(2);
  expect(rates.filter((row) => row.turns <= 2)).toHaveLength(21);
  expect(fixedStep(0.65, (p) => move([0, 0, 0], p, -1))).toBe(2);
  const results = Array.from({ length: 41 }, (_, i) => i - 20).flatMap((a) =>
    Array.from({ length: 41 }, (_, i) => keyboardTrial(a, i - 20)),
  );
  expect(results.filter(Boolean).length).toBeGreaterThan(1);
  expect(results.filter(Boolean).length / results.length).toBeLessThan(0.5);
});
it('keeps both perpendicular routes inside the contour band while the steep axis fails', () => {
  expect(contourError(start, [1, -1, 1])).toBeCloseTo(0.0005, 12);
  expect(contourError(start, [1, -1, -1])).toBeCloseTo(0.0005, 12);
  expect(contourError(start, [2, -1, 0])).toBeGreaterThan(0.001);
  expect(contourError(start, [1, 1, 0])).toBeGreaterThan(0.001);
  expect(contourError(start, [-1, 1, 0])).toBeGreaterThan(0.001);
});
