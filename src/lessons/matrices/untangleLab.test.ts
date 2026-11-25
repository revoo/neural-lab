// authored-by: Codex Meridian 6
import { describe, expect, it } from 'vitest';
import { corners, dot, lineMargin, margin, offset, search, transform, type Matrix, type Pair } from './untangleLab';

describe('Untangle the Data', () => {
  it('counts every matrix and every distinct horizontal projection with a shift', () => {
    const counts = search();
    expect(counts).toEqual({ total: 43046721, split: 5248800, area: 355886, shifted: 13702, shiftTrials: 531441 });
    for (const [pass, total] of [
      [counts.split, counts.total],
      [counts.area, counts.total],
      [counts.shifted, counts.shiftTrials],
    ]) {
      expect(pass).toBeGreaterThan(1);
      expect(pass / total).toBeLessThan(0.5);
    }
  });

  it('cannot split the offset cloud through the origin in any sampled direction', () => {
    for (let tick = 0; tick < 36000; tick++) {
      const theta = (tick * Math.PI) / 18000;
      const direction: Pair = [Math.cos(theta), Math.sin(theta)];
      if (margin(offset, (p) => dot(direction, p)) > 1e-9) throw Error('Separator at tick ' + tick);
    }
    // Exact certificate between samples: opposite labels on the same positive ray.
    expect(offset.positive).toContainEqual([2, 2]);
    expect(offset.negative).toContainEqual([0.5, 0.5]);
  });

  it('both XOR segments share their midpoint under every affine map', () => {
    const midpoint = ([a, b]: Pair[]) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    expect(midpoint(corners.positive)).toEqual(midpoint(corners.negative));
    // Affine maps preserve midpoints. Strictly separated segments cannot share one.
  });

  it('a kink separates XOR with several lines, while most tested placements fail', () => {
    const m: Matrix = [
      [1, -1],
      [-1, 1],
    ];
    const move = (p: Pair) => transform(p, m, [0, 0], true);
    expect(corners.positive.map(move)).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(corners.negative.map(move)).toEqual([
      [2, 0],
      [0, 2],
    ]);
    expect(
      lineMargin(corners, move, [
        [1, 0],
        [0, 1],
      ]),
    ).toBeCloseTo(Math.SQRT1_2);
    let successes = 0,
      tried = 0;
    for (let x = -4; x <= 4; x++)
      for (let y = -4; y <= 4; y++) {
        for (let u = -4; u <= 4; u++)
          for (let v = -4; v <= 4; v++) {
            if (
              lineMargin(corners, move, [
                [x, y],
                [u, v],
              ]) > 1e-9
            )
              successes++;
            tried++;
          }
      }
    expect(successes).toBeGreaterThan(1);
    expect(successes / tried).toBeLessThan(0.5);
  });
});
